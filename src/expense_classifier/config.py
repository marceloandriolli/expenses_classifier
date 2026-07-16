"""Configuração central: caminhos, categorias e limiares.

Precedência do diretório de dados (merchants.json, labels.csv, model.joblib):
  1. argumento --data-dir da CLI
  2. variável de ambiente EXPENSE_CLASSIFIER_HOME
  3. ~/.config/expense-classifier/  (padrão, estilo XDG)

No primeiro uso, merchants.json é semeado a partir do default embarcado
no pacote (data/merchants.default.json).
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

ENV_HOME = "EXPENSE_CLASSIFIER_HOME"

CATEGORIES: tuple[str, ...] = (
    "supermercado",
    "bares e restaurantes",
    "cafés e panificadoras",
    "moradia",
    "transporte",
    "lazer",
    "vestuário",
    "saúde",
    "serviços",
    "pet",
    "educação",
)

REVIEW_LABEL = "revisar"
IGNORE_LABEL = "ignorado"

ML_CONFIDENCE_THRESHOLD = 0.55
ML_MIN_EXAMPLES_PER_CLASS = 3


def default_data_dir() -> Path:
    if env := os.environ.get(ENV_HOME):
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_CONFIG_HOME", "~/.config")
    return Path(xdg).expanduser() / "expense-classifier"


@dataclass(frozen=True)
class Settings:
    """Caminhos resolvidos para uma execução."""

    data_dir: Path = field(default_factory=default_data_dir)

    @property
    def merchants_path(self) -> Path:
        return self.data_dir / "merchants.json"

    @property
    def labels_path(self) -> Path:
        return self.data_dir / "labels.csv"

    @property
    def model_path(self) -> Path:
        return self.data_dir / "model.joblib"

    def ensure_initialized(self) -> None:
        """Cria o diretório de dados e semeia merchants.json no primeiro uso."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if not self.merchants_path.exists():
            default = Path(__file__).parent / "data" / "merchants.default.json"
            shutil.copy(default, self.merchants_path)
        if not self.labels_path.exists():
            self.labels_path.write_text("description,category\n", encoding="utf-8")
