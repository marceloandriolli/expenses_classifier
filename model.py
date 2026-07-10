"""Fallback ML: TF-IDF char_wb(2,5) + LinearSVC calibrado.

Treinado por bootstrap: as regras determinísticas (merchant + keyword)
rotulam o que conseguem e viram exemplos de treino, junto com correções
manuais em labels.csv. O papel do modelo é generalizar variações que as
regras não cobrem (nomes truncados pelo banco, grafias alternativas).
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from config import CATEGORIES, ML_MIN_EXAMPLES_PER_CLASS, Settings
from normalize import normalize
from rules import is_ignorable, load_merchants, match_keyword, match_merchant

log = logging.getLogger(__name__)


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=1
                ),
            ),
            ("clf", CalibratedClassifierCV(LinearSVC(C=1.0), cv=3)),
        ]
    )


def collect_training_data(
    settings: Settings, df: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Bootstrap: rótulos das regras + merchants.json + labels.csv manual."""
    merchants = load_merchants(settings.merchants_path)
    rows: list[tuple[str, str]] = []

    if df is not None and "description" in df.columns:
        for desc in df["description"].dropna().unique():
            norm = normalize(desc)
            if not norm or is_ignorable(norm):
                continue
            cat = match_merchant(norm, merchants) or match_keyword(norm)
            if cat:
                rows.append((norm, cat))

    rows.extend(merchants.items())

    if settings.labels_path.exists():
        try:
            manual = pd.read_csv(settings.labels_path)
        except pd.errors.EmptyDataError:
            manual = pd.DataFrame(columns=["description", "category"])
        for _, r in manual.iterrows():
            cat = str(r.get("category", "")).strip().lower()
            if cat in CATEGORIES:
                rows.append((normalize(str(r["description"])), cat))
            elif cat:
                log.warning("labels.csv: categoria inválida ignorada: %r", cat)

    return pd.DataFrame(rows, columns=["text", "category"]).drop_duplicates()


def train(
    settings: Settings, df: pd.DataFrame | None = None
) -> Pipeline | None:
    """Treina e persiste o modelo. Retorna None se não houver dados suficientes."""
    data = collect_training_data(settings, df)
    counts = data["category"].value_counts()
    trainable = data[
        data["category"].isin(counts[counts >= ML_MIN_EXAMPLES_PER_CLASS].index)
    ]

    if trainable.empty or trainable["category"].nunique() < 2:
        log.warning("dados insuficientes para treinar o ML; usando apenas regras")
        return None

    pipeline = build_pipeline()
    pipeline.fit(trainable["text"], trainable["category"])
    _atomic_dump(pipeline, settings.model_path)
    log.info(
        "modelo treinado: %d exemplos, %d classes -> %s",
        len(trainable),
        trainable["category"].nunique(),
        settings.model_path,
    )
    return pipeline


def load(settings: Settings) -> Pipeline | None:
    if not settings.model_path.exists():
        return None
    try:
        return joblib.load(settings.model_path)
    except Exception:  # noqa: BLE001 — modelo corrompido/incompatível não é fatal
        log.warning(
            "falha ao carregar %s (versão de sklearn diferente?); "
            "retreine com `expense-classifier train`",
            settings.model_path,
        )
        return None


def _atomic_dump(obj: object, path: Path) -> None:
    """Escreve via arquivo temporário + rename: nunca deixa modelo pela metade."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        joblib.dump(obj, tmp)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
