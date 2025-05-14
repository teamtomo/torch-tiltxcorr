"""Cross correlation with cosine stretching for cryo-EM data in PyTorch"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torch-tiltxcorr")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Alister Burt"
__email__ = "alisterburt@gmail.com"

from torch_tiltxcorr.tiltxcorr import tiltxcorr, tiltxcorr_no_stretch

__all__ = ['tiltxcorr', 'tiltxcorr_no_stretch']
