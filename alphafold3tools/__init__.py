from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("alphafold3_tools")
except PackageNotFoundError:
    # if the package is not installed, use a default version
    __version__ = "0.0.0"
