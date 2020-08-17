"""Shortened configuration example.

pycnfg provides convenient features:
    * to increase configuration readability.
Specify verbose kwargs in separate sections ('gs_params' in example).
    * to reduce boilerplate.
Use ``global`` key on configuration, section and sub-configuration levels.

"""

import lightgbm
import mlshell
import pycnfg
import sklearn

target_transformer = sklearn.preprocessing.PowerTransformer(
    method='yeo-johnson', standardize=True, copy=True)


# Optimization hp ranges.
hp_grid = {
    # 'process_parallel__pipeline_numeric__transform_normal__skip': [False],
    # 'process_parallel__pipeline_numeric__scale_column_wise__quantile_range': [(1, 99)],
    'process_parallel__pipeline_numeric__add_polynomial__degree': [1, 2],
    'estimate__transformer': [None, target_transformer],

    # sgd
    # 'estimate__regressor__alpha': np.logspace(-2, -1, 10),
    # 'estimate__regressor__l1_ratio': np.linspace(0.1, 1, 10),
}


CNFG = {
    # In ``pycnfg.run`` set default configuration :data:`mlshell.CNFG`, that
    # has pre-defined path, logger sections and main sub-keys (see below).
    'pipeline': {
        'sgd': {
            'kwargs': {
                'estimator_type': 'regressor',
                'estimator': sklearn.linear_model.SGDRegressor(
                    penalty='elasticnet', l1_ratio=1, shuffle=False,
                    max_iter=1000, alpha=0.02, random_state=42),
            }
        },
        'lgbm': {
            'kwargs': {
                'estimator_type': 'regressor',
                'estimator': lightgbm.LGBMRegressor(
                    num_leaves=2, min_data_in_leaf=60, n_estimators=200,
                    max_depth=-1, random_state=42),
            }
        }
    },
    'metric': {
        'r2': {
            'score_func': sklearn.metrics.r2_score,
            'kwargs': {'greater_is_better': True},
        },
        'mse': {
            'score_func': sklearn.metrics.mean_squared_error,
            'kwargs': {
                'greater_is_better': True,
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
            'filepath': 'data/test.csv',
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
            'hp': hp_grid,
            'hp_grid': hp_grid,
            'gs_params': 'gs_params__conf',
            'metric_id': ['metric__r2', 'metric__mse'],
            'steps': [
                ('fit',),
                ('validate',),
                ('optimize',),
                ('validate',),
                ('predict',),
                ('dump',),
            ],
        },
    },
    # Separate section for 'gs_params' kwarg.
    'gs_params': {
        'conf': {
            'priority': 3,
            'init': {
                'n_iter': None,
                'n_jobs': 1,
                'refit': 'metric__r2',
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
