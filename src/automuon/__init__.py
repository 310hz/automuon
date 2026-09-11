from importlib.metadata import version as _version

from .optimizer import get_muon_and_adam
from .mark import with_muon, without_muon, with_adam, without_adam


__version__ = _version("automuon")
