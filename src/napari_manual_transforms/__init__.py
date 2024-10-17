from importlib import metadata

try:
    __version__ = metadata.version("napari-manual-transforms")
except metadata.PackageNotFoundError:
    # package is not installed
    __version__ = "uninstalled"

from ._widget import TransformationWidget

__all__ = ["TransformationWidget"]
