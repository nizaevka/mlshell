# coding: utf-8
"""MLshell, universe ML workflow."""


from .__version__ import __version__


import warnings


from .callbacks import find_path
from .logger import CreateLogger
from .read_conf import GetParams
from .general import Workflow
from .default import CreateDefaultPipeline, DEFAULT_PARAMS
from .gui import GUI


__all__ = ['find_path', 'CreateLogger', 'GetParams', 'Workflow', 'CreateDefaultPipeline', 'DEFAULT_PARAMS', 'GUI']

# import platform
# if platform.system() != 'Windows':
#     warnings.warn("Package was tested only on Windows os", UserWarning)