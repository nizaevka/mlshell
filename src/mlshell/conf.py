"""The module contains default configuration and pipeline steps."""


import mlshell.pycnfg as pycnfg
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
            ('create', {})
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
            ('create', {
                'cache': None,
                'steps': None,
                'estimator': sklearn.linear_model.LinearRegression(),
                'estimator_type': 'regressor',
                # 'th_strategy': None,
            },),
            ('resolve', {},),
                # [deprecated]  should be setted 'auto'/['auto'], by default only for index
                #  only if not setted
                # 'hp': {
                #     'process_parallel__pipeline_categoric__select_columns__kwargs',
                #     'process_parallel__pipeline_numeric__select_columns__kwargs',
                #     'estimate__apply_threshold__threshold'}
                # },
        ],
    },
}


METRICS = {
    'accuracy': {
        'init': None,
        'producer': mlshell.MetricProducer,
        'global': {},
        'patch': {},
        'priority': 3,
        'steps': [
            ('make_scorer', {
                'func': sklearn.metrics.accuracy_score,
                'kwargs': {'greater_is_better': True},
            }),
        ],
    },
    'r2': {
        'init': None,
        'producer': mlshell.MetricProducer,
        'global': {},
        'patch': {},
        'priority': 3,
        'steps': [
            ('make_scorer', {
                'func': sklearn.metrics.r2_score,
                'kwargs': {'greater_is_better': True},
            }),
        ],
    }
}


DATASETS = {
    'default': {
        'init': mlshell.Dataset(),
        'producer': mlshell.DataProducer,
        'global': {},
        'patch': {},
        'priority': 3,
        'steps': [
            ('load_cache', {'prefix': None},),
            ('load', {},),
            ('preprocess', {'categor_names': [], 'target_names': [], 'pos_labels': []},),
            ('info', {},),
            ('unify', {},),
            ('split', {},),
            ('dump_cache', {'prefix': None},),
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
                'pipeline_id': None,
                'dataset_id': 'train',
                'fit_params': {},
                'hp': {},
                'resolver': mlshell.Resolver,
                'resolve_params': {},
            },),
            ('optimize', {
                'optimizer': mlshell.RandomizedSearchOptimizer,  # optimizer
                'validator': mlshell.Validator,
                'resolver': mlshell.Resolver,
                'pipeline_id': None,  # multiple pipeline? no, user can defined separately if really needed
                'dataset_id': 'train',
                'hp_grid': {},
                'scoring': None,
                'gs_params': {
                   'n_iter': None,
                   'n_jobs': 1,
                   'refit': None,  # no resolving
                   'cv': sklearn.model_selection.KFold(n_splits=3, shuffle=True),
                   'verbose': 1,
                   'pre_dispatch': 'n_jobs',
                   # TODO: for thresholdoptimizer, also need add pass_custom step.
                   #   so here params to mock.
                   # 'th_name':
                },
                'fit_params': {},
                'resolve_params': {
                    'estimate__apply_threshold__threshold': {
                        'samples': 10,
                        'plot_flag': False,
                        'fit_params': {},
                        'cv': sklearn.model_selection.KFold(n_splits=3, shuffle=True),
                    },
                },
            },),
            ('dump', {'pipeline_id': None}),
            ('validate', {
                'dataset_id': 'train',
                'validator': None,
                'metric': None,
                'pos_label': None,  # if None, get -1
                'pipeline_id': None,
            },),
            ('plot', {
                'plotter': None,  # gui
                'pipeline_id': None,
                'hp_grid': {},
                'dataset_id': 'train',
                'base_sort': False,
                # TODO: [beta]
                # 'dynamic_metric': None,
            }),
            ('predict', {
                'dataset_id': 'test',
                'pipeline_id': None,
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

For ML task, common sections would be:
* create/read pipelines and datasets objects.
* create workflow class and call methods with pipeline/dataset as argument.

"""


if __name__ == '__main__':
    pass
