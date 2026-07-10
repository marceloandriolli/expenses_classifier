"""Cascata de classificação: ignore -> merchant -> keyword -> guarda -> ML -> revisar."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

import model as ml
from config import (
    IGNORE_LABEL,
    ML_CONFIDENCE_THRESHOLD,
    REVIEW_LABEL,
    Settings,
)
from normalize import normalize
from rules import (
    is_ignorable,
    load_merchants,
    looks_like_person,
    match_keyword,
    match_merchant,
)

log = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("description",)


@dataclass(frozen=True)
class Classification:
    category: str
    method: str  # ignore | merchant | keyword | ml | none
    confidence: float


class Classifier:
    """Carrega regras e modelo uma vez; classifica N descrições."""

    def __init__(self, settings: Settings, retrain_with: pd.DataFrame | None = None):
        settings.ensure_initialized()
        self.settings = settings
        self.merchants = load_merchants(settings.merchants_path)
        if retrain_with is not None:
            self.model = ml.train(settings, retrain_with)
        else:
            self.model = ml.load(settings)

    def classify_one(
        self, description: object, transaction_type: object = None
    ) -> Classification:
        norm = normalize(description)

        if not norm or is_ignorable(norm, transaction_type):
            return Classification(IGNORE_LABEL, "ignore", 1.0)

        if cat := match_merchant(norm, self.merchants):
            return Classification(cat, "merchant", 1.0)

        if cat := match_keyword(norm):
            return Classification(cat, "keyword", 0.9)

        if self.model is not None and not looks_like_person(norm):
            proba = self.model.predict_proba([norm])[0]
            best = int(proba.argmax())
            conf = float(proba[best])
            if conf >= ML_CONFIDENCE_THRESHOLD:
                return Classification(
                    str(self.model.classes_[best]), "ml", round(conf, 3)
                )

        return Classification(REVIEW_LABEL, "none", 0.0)

    def classify_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"CSV sem coluna(s) obrigatória(s): {', '.join(missing)}. "
                f"Colunas presentes: {', '.join(df.columns)}"
            )

        has_type = "transaction_type" in df.columns
        results = [
            self.classify_one(
                row.description, row.transaction_type if has_type else None
            )
            for row in df.itertuples(index=False)
        ]

        out = df.copy()
        out["description_normalized"] = df["description"].map(normalize)
        out["category"] = [r.category for r in results]
        out["method"] = [r.method for r in results]
        out["confidence"] = [r.confidence for r in results]
        return out
