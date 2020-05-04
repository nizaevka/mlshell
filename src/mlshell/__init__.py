# coding: utf-8
"""MLshell, universe ML workflow."""


from .__version__ import __version__
from .callbacks import find_path
from .logger import CreateLogger
from .read_conf import GetParams
from .data import DataFactory
from .pipeline import PipeFactory
from .general import Workflow
from .default import CreateDefaultPipeline, DEFAULT_PARAMS
from .gui import GUI
from .eda import EDA
from .run import run


__all__ = ['run', 'find_path', 'CreateLogger', 'GetParams', 'DataFactory', 'PipeFactory',
           'Workflow', 'CreateDefaultPipeline', 'DEFAULT_PARAMS', 'GUI', 'EDA']

# import platform
# import warnings
# if platform.system() != 'Windows':
#     warnings.warn("Package was tested only on Windows os", UserWarning)
