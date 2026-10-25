"""Resolve the application version for CLI output and saved batch identity."""

from importlib.metadata import PackageNotFoundError, version


def package_version() -> str:
    """Use installed metadata, retaining the fallback for source checkouts."""

    try:
        return version("uma-bct-cli")
    except PackageNotFoundError:
        return "0.1.0"
