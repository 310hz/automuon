from importlib.metadata import version

from .optimizer import get_muon_with_adam
from .mark import with_muon, without_muon, with_adam, without_adam

__version__ = version("automuon")
