"""The :mod:`mlshell.conf` contains default configuration."""


import pycnfg
import sklearn
import mlshell

__all__ = ['CNFG']


PATHS = {
    'default': {
        'init': pycnfg.find_path,
        'producer': pycnfg.Producer,
        'priority': 1,
        'steps': [],
    }
}


LOGGERS = {
    'default': {
        'init': 'default',
        'producer': mlshell.LoggerProducer,
        'priority': 2,
        'steps': [
            ('make',),
        ],
    }
}


PIPELINES = {
    'default': {
        'init': mlshell.Pipeline,
        'producer': mlshell.PipelineProducer,
        'priority': 3,
        'steps': [
            ('make', ),
        ],
    },
}


METRICS = {
    'default': {
        'init': mlshell.Metric,
        'producer': mlshell.MetricProducer,
        'priority': 3,
        'steps': [
            ('make', ),
        ],
    },
}


DATASETS = {
    'default': {
        'init': mlshell.Dataset,
        'producer': mlshell.DatasetProducer,
        'priority': 3,
        'steps': [
            ('load', ),
            ('info', ),
            ('preprocess', ),
            ('split',),
        ],
    },
}

WORKFLOWS = {
    'default': {
        'init': {},
        'producer': mlshell.Workflow,
        'global': {},
        'patch': {},
        'priority': 4,
        'steps': [
            ('optimize', ),
            ('validate', ),
            ('predict', ),
            ('dump', ),
        ],
    },
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
