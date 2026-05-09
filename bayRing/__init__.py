# -*- coding: utf-8 -*-

from pathlib import Path

try:
    from importlib.metadata import PackageNotFoundError, distribution
except ImportError:
    try:
        from importlib_metadata import PackageNotFoundError, distribution
    except ImportError:
        PackageNotFoundError = Exception
        distribution = None

__version__ = "0+unknown"
try:
    if distribution is None:
        raise PackageNotFoundError

    _dist = distribution("bayRing")
    _package_root = Path(__file__).resolve().parent
    _dist_root = Path(_dist.locate_file("")).resolve()
    try:
        _package_root.relative_to(_dist_root)
    except ValueError:
        raise PackageNotFoundError
    __version__ = _dist.version
except Exception:
    try:
        from setuptools_scm import get_version

        __version__ = get_version(root="..", relative_to=__file__)
    except Exception:
        pass
