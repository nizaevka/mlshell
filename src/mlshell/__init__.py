# coding: utf-8
"""MLshell, universe ML workflow."""


from .__version__ import __version__
from .producers.logger import LoggerProducer
from .producers.dataset import DatasetProducer, Dataset
from .producers.pipeline import PipelineProducer, Pipeline
from .default import PipelineSteps, DEFAULT_PARAMS
from .resolve import HpResolver
from .optimize import RandomizedSearchOptimizer, ThresholdOptimizer
from .validate import Validator
from .general import Workflow
from .gui import GUI
from .eda import EDA

__all__ = ['DatasetProducer', 'PipelineProducer', 'Workflow'
           'Dataset', 'Pipeline',
           'HpResolver', 'GUI', 'EDA',
           'PipelineSteps', 'DEFAULT_PARAMS',
           'RandomizedSearchOptimizer', 'ThresholdOptimizer', 'Validator']


# import platform
# import warnings
# if platform.system() = 'Windows':
#     warnings.warn("Package was tested only on UNIX os", UserWarning)
