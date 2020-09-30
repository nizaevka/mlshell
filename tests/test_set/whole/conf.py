"""
Pipeline without steps:
    no param resolving (no encoder, no gap filling)
* hp_grid(some,empty) + Optimizer
* hp_grid(some,empty) + MockOptimizer will raise Error "no steps".

Also
    @time_profiler
    @memory_profiler
"""

import atexit

import line_profiler
import mlshell
import pycnfg
import sklearn
from memory_profiler import profile as memory_profiler

time_profiler = line_profiler.LineProfiler()
atexit.register(time_profiler.print_stats, output_unit=1)


# Optimization hp ranges.
hp_grid_1 = {
    'alpha': [100, 1000],
    'l1_ratio': [0, 1],
}


CNFG = {
    'pipeline': {
        'sgd': {
            'kwargs': {
                'steps': [],  # Set direct.
                'estimator': sklearn.linear_model.SGDRegressor(
                    penalty='elasticnet', l1_ratio=1, shuffle=False,
                    alpha=0.02, early_stopping=True, random_state=42),
            }
        },
    },
    'metric': {
        'mse': {
            'score_func': sklearn.metrics.mean_squared_error,
            'kwargs': {
                'greater_is_better': False,
                'squared': False
            },
        },
    },
    'dataset': {
        # Section level 'global' to specify common kwargs for test and train.
        'global': {'targets_names': ['wage'],
                   'categor_names': ['union', 'goodhlth', 'black', 'female',
                                     'married', 'service'],
                   },
        'train': {
            'filepath': './data/train.csv',
            'split__kwargs': {'train_size': 0.75, 'shuffle': False},
        },
        'test': {
            'filepath': './data/test.csv',
            'split__kwargs': {'train_size': 1},
        },
    },
    'workflow': {
        'conf': {
            # Global values will replace kwargs in corresponding default steps
            # => easy switch between pipeline for example (pycnfg move unknown
            # keys to 'global' by default).
            'pipeline_id': 'pipeline__sgd',
            'dataset_id': 'dataset__train',
            'predict__dataset_id': 'dataset__test',
            'hp': hp_grid_1,
            'gs_params': 'gs_params__conf_1',
            'metric_id': ['metric__mse'],
            'steps': [
                ('fit',),
                ('validate',),
                ('optimize', {'hp_grid': hp_grid_1}, [time_profiler,
                                                      memory_profiler]),
                ('optimize', {'hp_grid': {}}),
                # raise AttributeError, needs steps.
                # ('optimize', {'hp_grid': hp_grid_1,
                #              'optimizer': mlshell.model_selection.MockOptimizer
                #               }),
                # ('optimize', {'hp_grid': {},
                #               'optimizer': mlshell.model_selection.MockOptimizer
                #               }),
                ('validate',),
                ('predict',),
                ('dump',),
            ],
        },
    },
    # Separate section for 'gs_params' kwarg.
    'gs_params': {
        'conf_1': {
            'priority': 3,
            'init': {
                'n_iter': None,
                'n_jobs': 1,
                'refit': 'metric__mse',
                'cv': sklearn.model_selection.KFold(n_splits=3,
                                                    shuffle=True,
                                                    random_state=42),
                'verbose': 1,
                'pre_dispatch': 'n_jobs',
                'return_train_score': True,
            },
        },
    },
}


if __name__ == '__main__':
    objects = pycnfg.run(CNFG, dcnfg=mlshell.CNFG)
