# coding: utf-8
"""Pycnfg, universal Python configuration."""


from .handler import Handler
from .producer import Producer
from .utils import find_path, run
from .default import DEFAULT

__all__ = ['run', 'find_path', 'Handler', 'Producer', 'DEFAULT']


import platform
import warnings
if platform.system() == 'Windows':
    warnings.warn("Package was tested only on UNIX os", UserWarning)
