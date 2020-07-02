# coding: utf-8
"""Pycnfg, universal Python configuration."""


import platform
import warnings

from .default import DEFAULT
from .handler import Handler
from .producer import Producer
from .utils import find_path, run

__all__ = ['run', 'find_path', 'Handler', 'Producer', 'DEFAULT', 'ID']


# Configuration id (fulfilled in Handel).
ID = None

if platform.system() == 'Windows':
    warnings.warn("Package was tested only on UNIX os", UserWarning)
