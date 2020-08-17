"""Classification model tuning example.

https://www.kaggle.com/c/ieee-fraud-detection

Current configuration:
* use 10000 rows subset of train and test.
* use lgbm.
* use target transformation y**0.25.
* optimize polynomial degree and 'yeo-johnson'.

"""

import lightgbm
import mlshell
import numpy as np
import pycnfg
import sklearn
import xgboost


# Set hp ranges for optimize.
hp_grid = {
    # 'pass_custom__kw_args': [{'param_a': 1, 'param_b': 'c'}, {'param_a': 2, 'param_b': 'd'}, ],
    # 'process_parallel__pipeline_numeric__impute__gaps__strategy': ['median', 'constant'],
    # 'process_parallel__pipeline_numeric__transform_normal__skip': [True],
    # 'process_parallel__pipeline_numeric__scale_column_wise__quantile_range': [(0, 100), (1, 99)],
    # 'process_parallel__pipeline_numeric__add_polynomial__degree': [1],

    # # lgbm
    # 'estimate__classifier__n_estimators': np.linspace(50, 1000, 10, dtype=int),
    'estimate__classifier__num_leaves': [2**i for i in range(1, 5 + 1)],
    # 'estimate__classifier__min_child_samples': scipy.stats.randint(1, 100),
    # 'estimate__classifier__max_depth': np.linspace(1, 30, 10, dtype=int),

    # 'estimate__apply_threshold__threshold': [0.1, 0.2]
}


# Custom metric example (precision with kwargs supporting)
def custom_score_metric(y_true, y_pred, **kw_args):
    """Custom precision metric."""
    if kw_args:
        # `pass_custom_kw_args` are passed here.
        # some logic.
        pass
    tp = np.count_nonzero((y_true == 1) & (y_pred == 1))
    fp = np.count_nonzero((y_true == 0) & (y_pred == 1))
    score = tp/(fp+tp) if tp+fp != 0 else 0
    return score


# set workflow params
params = {
    'data': {
        'split_train_size': 0.7,
        'del_duplicates': False,
        'train': {
            'args': ['data/train_transaction.csv',
                     'data/train_identity.csv'],
            'kw_args': {'rows_limit': 10000,
                        'random_skip': False,
                        'index_col': 'TransactionID'},
        },
        'test': {
            'args': ['data/test_transaction.csv',
                     'data/test_identity.csv'],
            'kw_args': {'rows_limit': 10000,
                        'random_skip': False,
                        'index_col': 'TransactionID'},
        },
    },
    'th': {
        'pos_label': 1,
        'strategy': 1,
        'samples': 10,
        'plot_flag': True,
    },
    'cache': {
        'pipeline': False,
        'unifier': False,
    },
    'seed': 42,
}

CNFG = {
    'pipeline': {
        'sgd': {
            'kwargs': {
                'estimator_type': 'classifier',
                'estimator': sklearn.linear_model.SGDClassifier(
                    penalty='elasticnet', l1_ratio=0.01, alpha=0.01,
                    shuffle=False, max_iter=1000, early_stopping=True,
                    learning_rate='invscaling', power_t=0.25, eta0=0.01,
                    verbose=1, random_state=42),
            }
        },
        'lgbm': {
            'kwargs': {
                'estimator_type': 'classifier',
                'estimator': lightgbm.LGBMRegressor(
                    objective='binary', n_estimators=500, num_leaves=32,
                    min_child_samples=1, max_depth=13, learning_rate=0.03,
                    boosting_type='gbdt', subsample_freq=3, subsample=0.9,
                    reg_alpha=0.3, reg_lambda=0.3, colsample_bytree=0.9,
                    silent=True, n_jobs=-1, random_state=42),
            }
        },
        'xgb': {
            'kwargs': {
                'estimator_type': 'classifier',
                'estimator': xgboost.XGBRegressor(
                    objective='binary:hinge', **{
                        'min_child_weight': 1, 'eta': 0.01,
                        'n_estimators': 100, 'colsample_bytree': 0.5,
                        'max_depth': 12, 'subsample': 0.8, 'alpha': 1,
                        'gamma': 1, 'silent': 1, 'verbose_eval': True,
                        'seed': 42,
                    }),
            }
        },
    },
    'metric': {
        'roc_auc': {
            'score_func': sklearn.metrics.roc_auc_score,
            'kwargs': {'greater_is_better': True, 'needs_proba': True},
        },
        'precision': {
            'score_func': sklearn.metrics.precision_score,
            'kwargs': {'greater_is_better': True, 'zero_division': 0,
                       'pos_label': 1}
        },
        'custom': {
            'score_func': custom_score_metric,
            'kwargs': {'greater_is_better': True, 'needs_custom_kwargs': True}
        },
        'confusion_matrix': {
            'score_func': sklearn.metrics.confusion_matrix,
            'kwargs': {'labels': [1, 0]}
        },
        'classification_report': {
            'score_func': sklearn.metrics.classification_report,
            'kwargs': {'output_dict': True, 'zero_division': 0}
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
            'metric_id': ['metric__roc_auc', 'metric__precision',
                          'metric__custom'],
        },
    },
    # Separate section for 'gs_params' kwarg.
    'gs_params': {
        'conf': {
            'priority': 3,
            'init': {
                'n_iter': None,
                'n_jobs': 1,
                'refit': 'metric__roc_auc',
                'cv': sklearn.model_selection.TimeSeriesSplit(n_splits=3),
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