"""Configuration example.

*
"""


import numpy as np
import sklearn
import lightgbm
import pycnfg


target_transformer = sklearn.preprocessing.PowerTransformer(method='yeo-johnson', standardize=True, copy=True)


# Optimization ho ranges.
hp_grid = {

    #'process_parallel__pipeline_numeric__transform_normal__skip': [False],
    #'process_parallel__pipeline_numeric__scale_column_wise__quantile_range': [(1, 99)],
    #'process_parallel__pipeline_numeric__add_polynomial__degree': [1],
    'estimate__transformer': [target_transformer],

    # sgd
    #'estimate__regressor__alpha': np.logspace(-2, -1, 10),
    #'estimate__regressor__l1_ratio': np.linspace(0.1, 1, 10),
}


CNFG = {
    'workflow': {
        'current': {
            'pipeline_id': 'pipeline__sgd',
            'dataset_id': 'train',
            'gs_params': 'gs_paramss__conf',
            'hp_grid': hp_grid,
            'scoring': ['r2'],
        },
    },
    'pipeline': {
        'sgd': {
            'estimator': sklearn.linear_model.SGDRegressor(
                penalty='elasticnet', l1_ratio=1, shuffle=False,
                max_iter=1000, alpha=0.02),
            'estimator_type': 'regressor',
        },
        'lgbm': {
            'estimator': lightgbm.LGBMRegressor(
                num_leaves=2, min_data_in_leaf=60,
                n_estimators=200, max_depth=-1),
            'estimator_type': 'regressor',
        }
    },
    'metric': {
        'r2': {
            'score_func': sklearn.metrics.r2_score,
            'greater_is_better': True,
        },
        'mse': {
            'score_func': sklearn.metrics.mean_squared_error,
            'greater_is_better': True,
            'squared': False
        },
    },
    'gs_params': {
        'conf': {
            'init': {
                'n_iter': None,
                'n_jobs': 1,
                'refit': ['r2'],
                'cv': sklearn.model_selection.KFold(n_splits=3, shuffle=True),
                'verbose': 1,
                'pre_dispatch': 'n_jobs',
                'return_train_score': True,
            }
        },
    },
    'dataset': {
        'train': {
            'steps': [
                ('load', {'filepath': './data/train.csv'}),
                ('info',),
                ('preprocess', {'target_names': ['targets'],
                                'categor_names': ['union', 'goodhlth',
                                                  'black', 'female',
                                                  'married', 'service']}),
                ('split', {'train_size': 0.75, 'test_size': None,
                           'shuffle': False, 'stratify': None,
                           'random_state': None}),
            ],
        },
        'test': {
            'steps': [
                ('load', {'filename': 'data/test.csv'}),
                ('info',),
                ('preprocess', {'categor_names': ['union', 'goodhlth',
                                                  'black', 'female',
                                                  'married', 'service'],
                                'target_names': ['targets']}),
            ],
        },
    },
}


if __name__ == '__main__':
    pycnfg.run(CNFG)
