"""expense-classifier: local BR bank-transaction expense classifier."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("expense-classifier")
except PackageNotFoundError:  # running from a source tree without install
    __version__ = "0.0.0+dev"

from .cascade import Classifier
from .config import Settings

__all__ = ["Classifier", "Settings", "__version__"]
