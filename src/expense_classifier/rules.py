"""Regras determinísticas da cascata: ignore, merchant lookup, keywords, guarda."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from .config import CATEGORIES
from .normalize import strip_accents

log = logging.getLogger(__name__)


class MerchantsFileError(ValueError):
    """merchants.json inválido (JSON malformado ou categoria desconhecida)."""


# ---------------------------------------------------------------------------
# Padrões a ignorar: não são despesa classificável
# ---------------------------------------------------------------------------

_IGNORE_PATTERNS = re.compile(
    r"^(PAGAMENTO RECEBIDO|PAGAMENTO DE FATURA|IOF DE VOLTA|ESTORNO|RENDIMENTO|"
    r"APLICACAO|RESGATE|TRANSFERENCIA RECEBIDA|PIX - RECEBIDO)",
    re.I,
)

_INCOME_TYPES = frozenset({"income", "receita", "credit"})


def is_ignorable(normalized: str, transaction_type: object = None) -> bool:
    if (
        transaction_type is not None
        and str(transaction_type).strip().lower() in _INCOME_TYPES
    ):
        return True
    return bool(_IGNORE_PATTERNS.match(normalized))


# ---------------------------------------------------------------------------
# Merchant lookup
# ---------------------------------------------------------------------------

def load_merchants(path: Path) -> dict[str, str]:
    """Carrega e valida merchants.json.

    Chaves são normalizadas (sem acento, maiúsculas) e ordenadas da mais
    longa para a mais curta, para que o match mais específico vença.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MerchantsFileError(f"{path}: JSON inválido — {exc}") from exc

    merchants: dict[str, str] = {}
    for key, category in raw.items():
        if key.startswith("_"):
            continue
        cat = str(category).strip().lower()
        if cat not in CATEGORIES:
            raise MerchantsFileError(
                f"{path}: categoria inválida {category!r} na chave {key!r}. "
                f"Válidas: {', '.join(CATEGORIES)}"
            )
        merchants[strip_accents(key).upper()] = cat

    log.debug("merchants.json: %d entradas carregadas", len(merchants))
    return dict(sorted(merchants.items(), key=lambda kv: -len(kv[0])))


def match_merchant(normalized: str, merchants: dict[str, str]) -> str | None:
    for key, category in merchants.items():
        if key in normalized:
            return category
    return None


# ---------------------------------------------------------------------------
# Keywords genéricas (fallback antes do ML)
# ---------------------------------------------------------------------------

_KEYWORD_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(SUPERMERCADO|MERCADO|ATACAD|HORTIFRUTI|ACOUGUE|SACOLAO|EMPORIO)\b"), "supermercado"),
    (re.compile(r"\b(RESTAURANTE|PIZZ?ARIA|BURGU?ER|LANCHE|CHURRASC|SUSHI|BAR\b|BOTEQUIM|CHOPP|PETISC|HAMBURG|SORVETERIA|GELATERIA)\b"), "bares e restaurantes"),
    (re.compile(r"\b(CAFE|CAFETERIA|PADARIA|PANIFICADORA|CONFEITARIA|DOCERIA|DOCES)\b"), "cafés e panificadoras"),
    (re.compile(r"\b(IMOVEIS|IMOBILIARIA|CONDOMINIO|ALUGUEL|ENERGIA|ELETRIC|SANEAMENTO|CASAN|AGUA)\b"), "moradia"),
    (re.compile(r"\b(POSTO|COMBUSTIVEL|ESTACIONAMENTO|PEDAGIO|UBER|MECANICA|AUTO ?PECAS|PNEUS|LAVACAO)\b"), "transporte"),
    (re.compile(r"\b(CINEMA|CINE|INGRESSO|PARQUE|EVENTOS|SHOW|TEATRO|GAME|JOGOS)\b"), "lazer"),
    (re.compile(r"\b(MODAS?|VESTUARIO|CALCADOS|ROUPAS|BOUTIQUE|CONFECCOES|RENNER|RIACHUELO|C&A|HERING)\b"), "vestuário"),
    (re.compile(r"\b(FARMACIA|DROGARIA|DROGASIL|CLINICA|LABORATORIO|HOSPITAL|ODONTO|MEDIC|PSICOLOG|FISIOTERAP|UNIMED)\b"), "saúde"),
    (re.compile(r"\b(TELEFONICA|TELECOM|INTERNET|CLARO|TIM|VIVO|SOFTWARE|TECNOLOGIA|ASSINATURA|CARTORIO|SEGURO)\b"), "serviços"),
    (re.compile(r"\b(PET ?SHOP|VETERINAR|RACAO|BANHO E TOSA)\b"), "pet"),
    (re.compile(r"\b(ESCOLA|COLEGIO|EDUCACAO|CURSO|FACULDADE|UNIVERSIDADE|CRECHE|ENSINO|LIVRARIA)\b"), "educação"),
)


def match_keyword(normalized: str) -> str | None:
    for pattern, category in _KEYWORD_RULES:
        if pattern.search(normalized):
            return category
    return None


# ---------------------------------------------------------------------------
# Guarda anti-pessoa-física: PIX para indivíduo não tem sinal textual;
# sem token corporativo, o ML nunca deve chutar.
# ---------------------------------------------------------------------------

_CORPORATE_TOKENS = re.compile(
    r"\b(LTDA|EIRELI|S\.?/?A|MEI?|CIA|COMERCIO|SERVICOS?|INSTITUICAO|"
    r"DISTRIBUIDOR|INDUSTRIA|RESTAURANTE|MERCADO|FARMACIA|POSTO|CLINICA|"
    r"CENTRO|LOJA|CASA|BAR|CAFE|SORVETERIA|PADARIA|ESCOLA|COLEGIO|"
    r"ASSOCIACAO|COOPERATIVA|FUNDACAO|EVENTOS|TECNOLOGIA|PAGAMENTO)\b"
)


def looks_like_person(normalized: str) -> bool:
    if _CORPORATE_TOKENS.search(normalized):
        return False
    words = normalized.split()
    return 2 <= len(words) <= 6 and all(w.isalpha() for w in words)
