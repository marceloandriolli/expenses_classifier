"""Testes do classificador. Rodar com: pytest"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from expense_classifier.cascade import Classifier
from expense_classifier.config import IGNORE_LABEL, REVIEW_LABEL, Settings
from expense_classifier.normalize import normalize
from expense_classifier.rules import (
    MerchantsFileError,
    is_ignorable,
    load_merchants,
    looks_like_person,
    match_keyword,
)

# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("PIX - ENVIADO   20/06 17:23 GIASSI & CIA LTDA", "GIASSI & CIA LTDA"),
        ("Transferência enviada|GIASSI &amp; CIA LTDA", "GIASSI & CIA LTDA"),
        ("PAG BOLETO      IBAGY IMOVEIS LTDA", "IBAGY IMOVEIS LTDA"),
        ("GAS             COPA ENERGIA DISTRIBUIDOR", "COPA ENERGIA DISTRIBUIDOR"),
        ("PAGTO UNIMED    UNIMED FLORIANÓPOLIS", "UNIMED FLORIANOPOLIS"),
        ("PIX - ENVIADO   03/06 19:54 ENIR LUCIA TIDRE 01776154", "ENIR LUCIA TIDRE"),
        ("Transferência enviada|57.127.659 EIMILI PEREIRA", "EIMILI PEREIRA"),
        ("Transferência enviada|64 795 469 Djonatha Felipe Da Motta", "DJONATHA FELIPE DA MOTTA"),
        ("Pagamento efetuado|Mediarte Jardins", "MEDIARTE JARDINS"),
    ],
)
def test_normalize_strips_bank_noise(raw: str, expected: str) -> None:
    assert normalize(raw) == expected


def test_normalize_non_string_is_empty() -> None:
    assert normalize(None) == ""
    assert normalize(float("nan")) == ""
    assert normalize(123) == ""


def test_normalize_collapses_whitespace_and_accents() -> None:
    assert normalize("  PANIFICADORA   SÃO   JOSÉ  ") == "PANIFICADORA SAO JOSE"


# ---------------------------------------------------------------------------
# ignore
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "desc",
    [
        "PAGAMENTO DE FATURA",
        "PAGAMENTO RECEBIDO",
        "IOF DE VOLTA DE ANTHROPIC",
        "ESTORNO COMPRA",
    ],
)
def test_ignorable_patterns(desc: str) -> None:
    assert is_ignorable(desc)


def test_income_transaction_type_is_ignored() -> None:
    assert is_ignorable("QUALQUER COISA", transaction_type="Income")
    assert not is_ignorable("GIASSI & CIA LTDA", transaction_type="Expense")


# ---------------------------------------------------------------------------
# person guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["ENIR LUCIA TIDRE", "MAIRA ANTUNES PAVAN", "CYRIO MELOTO DE SOUZA JUN"],
)
def test_person_names_detected(name: str) -> None:
    assert looks_like_person(name)


@pytest.mark.parametrize(
    "name",
    [
        "GIASSI & CIA LTDA",
        "SHPP BRASIL INSTITUICAO D",
        "COPA ENERGIA DISTRIBUIDOR",
        "RESTAURANTE GUSTOS",
    ],
)
def test_companies_not_flagged_as_person(name: str) -> None:
    assert not looks_like_person(name)


# ---------------------------------------------------------------------------
# keywords
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("desc", "cat"),
    [
        ("FARMACIA NOVA ESPERANCA", "saúde"),
        ("POSTO SHELL CENTRO", "transporte"),
        ("SORVETERIA DO ZE", "bares e restaurantes"),
        ("PADARIA PAO QUENTE", "cafés e panificadoras"),
        ("LIVRARIA CULTURA", "educação"),
        ("PET SHOP AMIGO FIEL", "pet"),
    ],
)
def test_keyword_fallback(desc: str, cat: str) -> None:
    assert match_keyword(desc) == cat


# ---------------------------------------------------------------------------
# merchants.json validation
# ---------------------------------------------------------------------------


def test_invalid_category_raises(tmp_path: Path) -> None:
    bad = tmp_path / "merchants.json"
    bad.write_text(json.dumps({"FOO": "categoria-inexistente"}), encoding="utf-8")
    with pytest.raises(MerchantsFileError):
        load_merchants(bad)


def test_malformed_json_raises(tmp_path: Path) -> None:
    bad = tmp_path / "merchants.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(MerchantsFileError):
        load_merchants(bad)


def test_longest_key_wins(tmp_path: Path) -> None:
    p = tmp_path / "merchants.json"
    p.write_text(
        json.dumps({"CASA": "moradia", "CASA DE CARNES": "supermercado"}),
        encoding="utf-8",
    )
    merchants = load_merchants(p)
    keys = list(merchants)
    assert keys.index("CASA DE CARNES") < keys.index("CASA")


# ---------------------------------------------------------------------------
# cascade (integração)
# ---------------------------------------------------------------------------


@pytest.fixture()
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("EXPENSE_CLASSIFIER_HOME", str(tmp_path / "data"))
    s = Settings(data_dir=tmp_path / "data")
    s.ensure_initialized()
    return s


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "description": [
                "PIX - ENVIADO   20/06 17:23 GIASSI & CIA LTDA",
                "PAGTO UNIMED    UNIMED FLORIANÓPOLIS",
                "Transferência enviada|ENIR LUCIA TIDRE 01776154940",
                "Pagamento de fatura",
                "PIX - ENVIADO   28/03 11:10 VETVIDA",
                "FARMACIA QUALQUER LTDA",
            ],
            "amount": [-100.0, -500.0, -50.0, -2000.0, -80.0, -30.0],
            "transaction_type": ["Expense"] * 6,
        }
    )


def test_cascade_end_to_end(settings: Settings, sample_df: pd.DataFrame) -> None:
    clf = Classifier(settings, retrain_with=sample_df)
    out = clf.classify_dataframe(sample_df)

    assert list(out["category"]) == [
        "supermercado",
        "saúde",
        REVIEW_LABEL,  # pessoa física: nunca chutar
        IGNORE_LABEL,  # fatura
        "pet",
        "saúde",  # keyword fallback
    ]
    assert set(out.columns) >= {
        "description_normalized",
        "category",
        "method",
        "confidence",
    }


def test_person_never_classified_by_ml(settings: Settings, sample_df: pd.DataFrame) -> None:
    clf = Classifier(settings, retrain_with=sample_df)
    result = clf.classify_one("Transferência enviada|FULANO DE TAL DA SILVA")
    assert result.category == REVIEW_LABEL
    assert result.method == "none"


def test_missing_description_column_raises(settings: Settings) -> None:
    clf = Classifier(settings)
    with pytest.raises(ValueError, match="description"):
        clf.classify_dataframe(pd.DataFrame({"foo": [1]}))


def test_first_run_seeds_merchants(settings: Settings) -> None:
    assert settings.merchants_path.exists()
    merchants = json.loads(settings.merchants_path.read_text(encoding="utf-8"))
    assert "GIASSI" in merchants


def test_output_preserves_input_columns(settings: Settings, sample_df: pd.DataFrame) -> None:
    clf = Classifier(settings)
    out = clf.classify_dataframe(sample_df)
    for col in sample_df.columns:
        assert col in out.columns
    assert len(out) == len(sample_df)
