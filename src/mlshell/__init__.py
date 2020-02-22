# coding: utf-8
"""MLshell, Light ML workflow."""

from .__version__ import __version__

import warnings
from platform import system


from .callbacks import find_path
from .logger import CreateLogger
from .general import Workflow
from .default import CreateDefaultPipeline, default_params
from .gui import GUI


__all__ = ['find_path', 'CreateLogger', 'Workflow', 'CreateDefaultPipeline', 'default_params', 'GUI']


if system() != 'Windows':
    warnings.warn("Package was tested only on Windows os", UserWarning)
