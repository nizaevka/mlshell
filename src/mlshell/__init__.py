"""MLshell - universe machine learning workflow."""


from .__version__ import __version__
from .producers.logger import LoggerProducer
from .producers.dataset import DatasetProducer, Dataset
from .producers.pipeline import PipelineProducer, Pipeline
from .producers.metric import MetricProducer, Metric
from .producers.workflow import Workflow

from .blocks import pipeline
from .blocks import plot
from .blocks import model_selection
from .blocks import preprocessing

from .conf import CNFG

__all__ = ['DatasetProducer', 'PipelineProducer', 'Workflow', 'MetricProducer',
           'Dataset', 'Pipeline', 'Metric',
           'pipeline', 'plot', 'model_selection', 'preprocessing',
           'CNFG']
