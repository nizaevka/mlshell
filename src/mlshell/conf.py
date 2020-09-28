"""The :mod:`mlshell.conf` contains default configuration."""


import pycnfg
import sklearn
import mlshell

__all__ = ['CNFG']


PATHS = {
    'init': pycnfg.find_path,
    'producer': pycnfg.Producer,
    'priority': 1,
    'steps': [],
    'default': {},
}


LOGGERS = {
    'init': 'default',
    'producer': mlshell.LoggerProducer,
    'priority': 2,
    'steps': [
        ('make',),
    ],
    'default': {},
}


PIPELINES = {
    'init': mlshell.Pipeline,
    'producer': mlshell.PipelineProducer,
    'priority': 3,
    'steps': [
        ('make', ),
    ],
}


METRICS = {
    'init': mlshell.Metric,
    'producer': mlshell.MetricProducer,
    'priority': 4,
    'steps': [
        ('make', ),
    ],
}


DATASETS = {
    'init': mlshell.Dataset,
    'producer': mlshell.DatasetProducer,
    'priority': 5,
    'steps': [
        ('load', ),
        ('info', ),
        ('preprocess', ),
        ('split',),
    ],
}

WORKFLOWS = {
    'init': {},
    'producer': mlshell.Workflow,
    'priority': 6,
    'steps': [
        ('optimize', ),
        ('validate', ),
        ('predict', ),
        ('dump', ),
    ],
}


CNFG = {
    'path': PATHS,
    'logger': LOGGERS,
    'pipeline': PIPELINES,
    'dataset': DATASETS,
    'metric': METRICS,
    'workflow': WORKFLOWS,
}
"""Default sections for ML task."""


if __name__ == '__main__':
    pass
