from importlib.metadata import version as _version

from .mark import with_adam, with_muon, without_adam, without_muon
from .optimizer import get_muon_and_adam

__version__ = _version("automuon")

__all__ = [
    "with_adam",
    "with_muon",
    "without_adam",
    "without_muon",
    "get_muon_and_adam",
]
