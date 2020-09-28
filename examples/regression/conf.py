"""Regression model tuning example.

https://www.kaggle.com/c/allstate-claims-severity

Ideas:

* estimator: linear vs lightgbm vs xgboost.
* features 'yeo-johnson' transformation.
* features quantile scaler.
* polynomial feature generation: degree 1 vs 2.
* target transformer: None vs np.log vs y**0.25.
* loss: mse vs mae vs `ln(cosh(x)).
* estimator hyper-parameters.

Current configuration:

* use 10000 rows subset of train and test.
* use lgbm with 'fair' objective.
* use target y**0.25 transformation and features 'yeo-johnson' transformation.
* optimize polynomial degree and 'yeo-johnson'.

"""

import lightgbm
import mlshell
import numpy as np
import pycnfg
import sklearn
import xgboost


def target_func(y):
    return y**0.25


def target_inverse_func(y):
    return y**4


# Create target transformers (avoid lambda function).
target_transformer = sklearn.preprocessing.FunctionTransformer(
    func=target_func, inverse_func=target_inverse_func,)
target_transformer_2 = sklearn.preprocessing.FunctionTransformer(
    func=np.log, inverse_func=np.exp)


# Set hp ranges for optimize.
hp_grid = {
    'process_parallel__pipeline_numeric__transform_normal__skip': [True, False],
    'process_parallel__pipeline_numeric__add_polynomial__degree': [1, 2],
    'estimate__transformer': [target_transformer],

    # lgbm
    # 'estimate__regressor__n_estimators': np.linspace(50, 1000, 10, dtype=int),
    # 'estimate__regressor__num_leaves': [2**i for i in range(1, 6 + 1)],
    # 'estimate__regressor__min_data_in_leaf': np.linspace(1, 100, 10, dtype=int),
    # 'estimate__regressor__min_data_in_leaf': scipy.stats.randint(1, 100),
    # 'estimate__regressor__max_depth': np.linspace(1, 30, 10, dtype=int),
}


CNFG = {
    'pipeline': {
        'sgd': {
            'estimator': sklearn.linear_model.SGDRegressor(
                penalty='elasticnet', l1_ratio=0.01, alpha=0.01,
                shuffle=False, max_iter=1000, early_stopping=True,
                learning_rate='invscaling', power_t=0.25, eta0=0.01,
                verbose=1, random_state=42),
        },
        'lgbm': {
            'estimator': lightgbm.LGBMRegressor(
                objective='fair', num_leaves=2, min_data_in_leaf=1,
                n_estimators=250, max_depth=-1, silent=False,
                random_state=42),
        },
        'xgb': {
            'estimator': xgboost.XGBRegressor(
                objective="reg:squarederror", **{
                    'min_child_weight': 1, 'eta': 0.01,
                    'n_estimators': 500, 'colsample_bytree': 0.5,
                    'max_depth': 12, 'subsample': 0.8, 'alpha': 1,
                    'gamma': 1, 'silent': 1, 'verbose_eval': True,
                    'seed': 42,
                }),
        },
    },
    'metric': {
        'r2': {
            'score_func': sklearn.metrics.r2_score,
            'kwargs': {'greater_is_better': True},
        },
        'mae': {
            'score_func': sklearn.metrics.mean_absolute_error,
            'kwargs': {'greater_is_better': False},
        },
    },
    'dataset': {
        # Section level 'global' to specify common kwargs for test and train.
        'global': {'targets_names': ['loss'],
                   'categor_names': [f'cat{i}' for i in range(1, 117)],
                   'load__kwargs': {'nrows': 10000, 'index_col': 'id'},
                   },
        'train': {
            'filepath': './data/train.csv',
            'split__kwargs': {'train_size': 0.7, 'shuffle': False},
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
            'pipeline_id': 'pipeline__lgbm',
            'dataset_id': 'dataset__train',
            'predict__dataset_id': 'dataset__test',
            'hp_grid': hp_grid,
            'gs_params': 'gs_params__conf',
            'metric_id': ['metric__mae', 'metric__r2'],
            'steps': [
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
                'refit': 'metric__mae',
                'cv': sklearn.model_selection.KFold(n_splits=3,
                                                    shuffle=True,
                                                    random_state=42),
                'verbose': 1000,
                'pre_dispatch': 'n_jobs',
                'return_train_score': True,
            },
        },
    },
}


if __name__ == '__main__':
    # Use default configuration :data:`mlshell.CNFG`, that has pre-defined path
    # logger sections and main sub-keys (see below)
    objects = pycnfg.run(CNFG, dcnfg=mlshell.CNFG)
