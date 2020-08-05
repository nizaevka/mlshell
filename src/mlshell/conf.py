"""The :mod:`mlshell.conf` contains default configuration."""


import pycnfg
import sklearn
import mlshell

__all__ = ['CNFG']


PATHS = {
    'default': {
        'init': pycnfg.find_path,
        'producer': pycnfg.Producer,
        'global': {},
        'patch': {},
        'priority': 1,
        'steps': [],
    }
}


LOGGERS = {
    'default': {
        'init': 'default',
        'producer': mlshell.LoggerProducer,
        'global': {},
        'patch': {},
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
        'global': {},
        'patch': {},
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
        'global': {},
        'patch': {},
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
        'global': {},
        'patch': {},
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
            ('fit', ),
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
