# -*- coding: utf-8 -*-

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("bayRing")
except PackageNotFoundError:
    pass
