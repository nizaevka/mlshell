"""The module to set user configuration parammeters.

Attributes:
    params: user parameters
        see default module for detailed description of parameters
"""


from mlshell.libs import np, sklearn
import lightgbm
import xgboost
import scipy


# 'regressor'
estimator = [
    lightgbm.LGBMRegressor(num_leaves=2, min_data_in_leaf=1, n_estimators=250, max_depth=-1, silent=False)
][0]


def target_func(y):
    return y**0.25


def target_inverse_func(y):
    return y**4


target_transformer = sklearn.preprocessing.FunctionTransformer(func=np.log, inverse_func=np.exp,
                                                               validate=False, check_inverse=True)

hp_grid = {
    'process_parallel__pipeline_numeric__add_polynomial__degree': [3],
    'estimate__transformer': [None, target_transformer],

    # # lgbm
    # 'estimate__regressor__n_estimators': np.linspace(50, 500, 2, dtype=int),
    # 'estimate__regressor__num_leaves':[2**i for i in range(1, 2 + 1)],
    # 'estimate__regressor__min_data_in_leaf': scipy.stats.randint(1, 3),
    # 'estimate__regressor__min_data_in_leaf':np.linspace(10, 100, 2, dtype=int),
    # 'estimate__regressor__max_depth':np.linspace(1, 5, 5, dtype=int),
    # 'estimate__regressor__objective': ['regression', 'regression_l1', 'fair'],
}


params = {
    'pipeline': {
        'estimator': estimator,
        'type': 'regressor',
        'fit_params': {},
        'steps': None,
        'debug': False,
    },
    'metrics': {
        'score': (sklearn.metrics.mean_absolute_error, {'greater_is_better': False}),
        'r2': (sklearn.metrics.r2_score, {'greater_is_better': True}),
    },
    'gs': {
        'flag': True,
        'splitter': sklearn.model_selection.KFold(n_splits=3, shuffle=True),
        'hp_grid': hp_grid,
        'verbose': 1000,
        'n_jobs': 1,
        'runs': None,
        'metrics': ['score', 'r2'],
    },
    'data': {
        'train': {
            'args': ['data/train_10k.csv'],
            'kw_args': {'rows_limit': 1000,
                        'random_skip': False,
                        'index_col': 'id'},
        },
        'test': {
            'args': ['data/test_10k.csv'],
            'kw_args': {'rows_limit': 1000,
                        'random_skip': False,
                        'index_col': 'id'},
        },
        'split_train_size': 0.7,
        'del_duplicates': False,
    },
    'cache': {
        'pipeline': False,
        'unifier': False,
    },
}
