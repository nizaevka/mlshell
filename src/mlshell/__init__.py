# coding: utf-8
"""MLshell, universe ML workflow."""


from .__version__ import __version__
from .default import PipelineSteps, DEFAULT_PARAMS
from .producers.logger import LoggerProducer
from .producers.dataset import DatasetProducer, Dataset
from .producers.pipeline import PipelineProducer, Pipeline
from .producers.metric import MetricProducer, Metric
from .producers.workflow import Workflow
from .blocks.resolver import Resolver
from .blocks.validator import Validator
from .blocks.optimizer import RandomizedSearchOptimizer, ThresholdOptimizer
from .blocks.plotter import Plotter
from .blocks.eda import EDA

__all__ = ['DatasetProducer', 'PipelineProducer', 'Workflow', 'MetricProducer',
           'Dataset', 'Pipeline', 'Metric',
           'Resolver', 'Validator', 'Plotter', 'EDA',
           'PipelineSteps', 'DEFAULT_PARAMS',
           'RandomizedSearchOptimizer', 'ThresholdOptimizer']


# import platform
# import warnings
# if platform.system() = 'Windows':
#     warnings.warn("Package was tested only on UNIX os", UserWarning)
