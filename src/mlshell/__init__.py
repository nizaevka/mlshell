"""MLshell - universe machine learning workflow."""


from .__version__ import __version__
from .producers.logger import LoggerProducer, LOGGER_CONFIG
from .producers.dataset import DatasetProducer, Dataset
from .producers.pipeline import PipelineProducer, Pipeline
from .producers.metric import MetricProducer, Metric
from .producers.workflow import Workflow

from .blocks import pipeline
from .blocks import plot
from .blocks import model_selection
from .blocks import preprocessing
from .blocks import decorator
from .blocks import decomposition

from .conf import CNFG

from pycnfg import run  # alias to pycnfg.run

__all__ = ['DatasetProducer', 'PipelineProducer', 'MetricProducer',
           'LoggerProducer', 'Workflow',
           'Dataset', 'Pipeline', 'Metric',
           'pipeline', 'plot', 'model_selection', 'preprocessing', 'decorator',
           'decomposition',
           'CNFG', 'LOGGER_CONFIG', 'run']
