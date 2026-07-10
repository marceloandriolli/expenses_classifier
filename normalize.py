"""Normalização de descrições de transação bancária.

Transforma a descrição crua do extrato (BB, Nubank e afins) em um nome de
merchant limpo: maiúsculo, sem acentos, sem prefixos de banco, sem
CNPJ/CPF parcial e sem timestamps.
"""

from __future__ import annotations

import html
import re
import unicodedata

# Prefixos de banco a remover (Banco do Brasil + estilo Nubank)
_PREFIX_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^PIX\s*-\s*(ENVIADO|RECEBIDO)\s+\d{2}/\d{2}\s+\d{2}:\d{2}\s*", re.I),
    re.compile(
        r"^(TRANSFERENCIA ENVIADA|TRANSFERENCIA RECEBIDA|"
        r"PAGAMENTO EFETUADO|PAGAMENTO AGENDADO)\s*\|\s*",
        re.I,
    ),
    re.compile(
        r"^(PAG BOLETO|PAGTO\s+\w+|PAGTO|GAS|TELEFONE|AGUA|LUZ|DEB\.?\s*AUTOM\.?)\s{2,}",
        re.I,
    ),
)

# CNPJ/CPF parcial no início ("57.127.659 EIMILI PEREIRA", "64 795 469 ...")
_LEADING_DIGITS = re.compile(r"^[\d .\-/]{6,}\s+")
# dígitos soltos no fim ("ENIR LUCIA TIDRE 01776154")
_TRAILING_DIGITS = re.compile(r"\s+[\d.\-/]{4,}$")
_MULTISPACE = re.compile(r"\s{2,}")


def strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )


def normalize(description: object) -> str:
    """Descrição crua -> merchant limpo, maiúsculo, sem acento.

    Aceita qualquer objeto (células NaN de pandas viram string vazia).
    """
    if not isinstance(description, str):
        return ""
    text = html.unescape(description).strip()
    text = strip_accents(text).upper()
    for pat in _PREFIX_PATTERNS:
        text = pat.sub("", text)
    text = _LEADING_DIGITS.sub("", text)
    text = _TRAILING_DIGITS.sub("", text)
    return _MULTISPACE.sub(" ", text).strip(" |-.")
