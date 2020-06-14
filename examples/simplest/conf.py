"""The module to set user configuration parammeters.

Attributes:
    params: user parameters.
        see mlshell.default module for detailed description of parameters.
"""

import numpy as np
import sklearn
import lightgbm
import mlshell


# choose estimator
estimator = [
    sklearn.linear_model.SGDRegressor(penalty='elasticnet', l1_ratio=1, shuffle=False, max_iter=1000, alpha=0.02)
    # lightgbm.LGBMRegressor(num_leaves=2, min_data_in_leaf=60, n_estimators=200, max_depth=-1),
    ][0]


target_transformer = sklearn.preprocessing.PowerTransformer(method='yeo-johnson', standardize=True, copy=True)


# set ranges for hp
hp_grid = {

    #'process_parallel__pipeline_numeric__transform_normal__skip': [False],
    #'process_parallel__pipeline_numeric__scale_column_wise__quantile_range': [(1, 99)],
    #'process_parallel__pipeline_numeric__add_polynomial__degree': [1],
    'estimate__transformer': [target_transformer],

    # sgd
    #'estimate__regressor__alpha': np.logspace(-2, -1, 10),
    #'estimate__regressor__l1_ratio': np.linspace(0.1, 1, 10),
}


# find project path/script name
project_path, script_name = mlshell.find_path()
# create logger
logger = mlshell.logger.CreateLogger(project_path, script_name).logger


# set workflow params
conf = {
    'workflow': {
        'workflow_1': {
            'global': {'gs_params': 'my_gs', 'pipeline_id': 'pipeline_1'},
        },
    },
    'pipeline': {
        'pipeline_1': {
            'global': {'estimator': estimator, 'type': 'regressor'},
        },
    },
    'metric': {
        'score': {
            'global': {
                'func': sklearn.metrics.r2_score,
                'kwargs': {'greater_is_better': True},
            },
        },
        'mae': {
            'global': {
                'func': sklearn.metrics.mean_absolute_error,
                'kwargs': {'greater_is_better': True},
            },
        },
        'mse': {
            'global': {
                'func': sklearn.metrics.mean_squared_error,
                'kwargs': {'greater_is_better': False, 'squared': False},
            },
        },
    },
    'gs_params': {
        'my_gs': {
            'init': {
                'hp_grid': hp_grid,
                'n_iter': None,  # 'gs__runs'
                'scoring': ['score', 'mae', 'mse'],
                'n_jobs': 1,
                'refit': 'score',
                'cv': sklearn.model_selection.KFold(n_splits=3, shuffle=True),  # gs__splitter
                'verbose': 1,
                'pre_dispatch': 1,
                'random_state': None,
                'error_score': np.nan,
                'return_train_score': True,
            },
        },
    },
    'dataset': {
        'train': {
            'steps': [
                # ('load_cache', {'func': None, 'prefix': 'train'}),
                ('get', {'func': None, 'filename': 'data/train.csv'}),
                ('preprocess', {'func': None, 'categor_names': [], 'target_names': ['targets']}),
                ('info', {'func': None}),
                ('unify', {'func': None}),
                ('split', {'func': None,
                           'train_size': 0.7,  # 'split_train_size': 0.75,
                           'test_size': None,
                           'shuffle': False,
                           'stratify': None,
                           'random_state': None,}),
                ('dump_cache', {'func': None, 'prefix': 'train'}),
            ],
        },
        'test': {
            'steps': [
                ('get', {'func': None, 'filename': 'data/test.csv'}),
                ('preprocess', {'func': None, 'categor_names': [], 'target_names': ['targets']}),
                ('unify', {'func': None}),
            ],
        },
    },
}


if __name__ == '__main__':
    mlshell.run(conf)
