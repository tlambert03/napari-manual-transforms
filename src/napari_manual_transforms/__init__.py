from importlib import metadata

try:
    __version__ = metadata.version("napari-manual-transforms")
except metadata.PackageNotFoundError:
    # package is not installed
    __version__ = "uninstalled"

from ._widget import TransformationWidget

__all__ = ["TransformationWidget"]


def __getattr__(name: str) -> object:
    if name == "RotationWidget":
        import warnings

        warnings.warn(
            "'RotationWidget' has been renamed to 'TransformationWidget'.",
            DeprecationWarning,
            stacklevel=2,
        )
        return TransformationWidget

    raise AttributeError(f"module {__name__} has no attribute {name}")
