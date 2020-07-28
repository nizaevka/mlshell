"""The module contains default configuration and pipeline steps."""


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
            ('make', {'logger_name': 'default'}),
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
            ('make', {
                'estimator_type': 'regressor',
                'estimator': sklearn.linear_model.LinearRegression(),
                'th_step': False,
            }),
        ],
    },
}


METRICS = {
    'accuracy': {
        'init': mlshell.Metric,
        'producer': mlshell.MetricProducer,
        'global': {},
        'patch': {},
        'priority': 3,
        'steps': [
            ('make', {
                'score_func': sklearn.metrics.accuracy_score,
                'greater_is_better': True,
            }),
        ],
    },
    'r2': {
        'init': mlshell.Metric,
        'producer': mlshell.MetricProducer,
        'global': {},
        'patch': {},
        'priority': 3,
        'steps': [
            ('make', {
                'score_func': sklearn.metrics.r2_score,
                'greater_is_better': True,
            }),
        ],
    }
}


DATASETS = {
    'default': {
        'init': mlshell.Dataset,
        'producer': mlshell.DatasetProducer,
        'global': {},
        'patch': {},
        'priority': 3,
        'steps': [
            ('load', {'filepath': './data/train.csv'}),
            ('preprocess', {'target_names': ['targets']}),
            ('info', {}),
            ('split', {}),
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
            ('fit', {
                'pipeline_id': 'default',
                'dataset_id': 'default',
                'subset_id': 'train',
                'hp': {},
                'resolver': mlshell.model_selection.Resolver,
                'resolve_params': {},
                'fit_params': {},
            }),
            ('optimize', {
                'pipeline_id': 'default',
                'dataset_id': 'default',
                'subset_id': 'train',
                'hp_grid': {},
                'scoring': ['r2'],
                'fit_params': {},
                'resolver': mlshell.model_selection.Resolver,
                'resolve_params': {
                    'estimate__apply_threshold__threshold': {
                        'cross_val_predict': {
                            'method': 'predict_proba',
                            'cv': sklearn.model_selection.KFold(n_splits=3, shuffle=True),
                            'fit_params': {},
                        },
                        'calc_th_range': {
                            'metric': None,
                            'sampler': None,
                            'samples': 10,
                            'plot_flag': False,
                        },
                    },
                },
                'optimizer': mlshell.model_selection.RandomizedSearchOptimizer,
                'gs_params': {
                   'n_iter': None,
                   'n_jobs': 1,
                   'refit': ['r2'],
                   'cv': sklearn.model_selection.KFold(n_splits=3, shuffle=True),
                   'verbose': 1,
                   'pre_dispatch': 'n_jobs',
                },
                'dirpath': None,
                'dump_params': {},
            }),
            ('validate', {
                'pipeline_id': 'default',
                'dataset_id': 'default',
                'subset_id': ['train', 'test'],
                'metric_id': ['r2'],
                'validator': None,
            }),
            ('dump', {'pipeline_id': 'default', 'dirpath': None}),
            ('predict', {
                'pipeline_id': None,
                'dataset_id': 'default',
                'subset_id': 'test',
                'dirpath': None,
            }),
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
"""Default sections for ML task.

For ML task, typical configuration:
* Specify metrics.
* Make or load pipelines / datasets objects.
* Produce results (workflow), calling pipeline/dataset/metric methods.

"""


if __name__ == '__main__':
    pass
