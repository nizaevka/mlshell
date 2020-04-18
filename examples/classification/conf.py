"""The module to set user configuration parammeters.

Attributes:
    params: user parameters.
        see mlshell.default module for detailed description of parameters.

"""

import numpy as np
import sklearn
import lightgbm
import xgboost
import scipy


# choose estimator
estimator = [
    # sklearn.linear_model.SGDClassifier(loss='hinge',
    #                                    penalty='elasticnet',
    #                                    l1_ratio=0.01,
    #                                    alpha=0.01,
    #                                    shuffle=False,
    #                                    max_iter=1000,
    #                                    early_stopping=True,
    #                                    learning_rate='invscaling',
    #                                    power_t=0.25,
    #                                    eta0=0.01,
    #                                    verbose=1),
    # xgboost.XGBClassifier(objective='binary:hinge', **{
    #     'min_child_weight': 1,
    #     'eta': 0.01,
    #     'n_estimators': 100,
    #     'colsample_bytree': 0.5,
    #     'max_depth': 12,
    #     'subsample': 0.8,
    #     'alpha': 1,
    #     'gamma': 1,
    #     'silent': 1,
    #     'verbose_eval': True,
    #     'seed': 42,
    # }),
    lightgbm.LGBMClassifier(objective='binary',
                            n_estimators=500,
                            num_leaves=32,
                            min_child_samples=1,
                            max_depth=13,
                            learning_rate=0.03,
                            boosting_type='gbdt',
                            subsample_freq=3,
                            subsample=0.9,
                            reg_alpha=0.3,
                            reg_lambda=0.3,
                            colsample_bytree=0.9,
                            silent=True,
                            n_jobs=-1,
                            )

][0]


# define hyperparameters (hps) to cv
# set ranges for hps
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


# classifier custom metric example
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
    'pipeline': {
        'estimator': estimator,
        'type': 'classifier',
        'fit_params': {},
        'steps': None,
        'debug': False,
    },
    'metrics': {
        'score': (sklearn.metrics.roc_auc_score, {'greater_is_better': True, 'needs_proba': True}),
        'precision': (sklearn.metrics.precision_score, {'greater_is_better': True, 'zero_division': 0, 'pos_label': 1}),
        'custom': (custom_score_metric, {'greater_is_better': True, 'needs_custom_kw_args': True}),
        'confusion matrix': (sklearn.metrics.confusion_matrix, {'labels': [1, 0]}),
        'classification report': (sklearn.metrics.classification_report, {'output_dict': True, 'zero_division': 0}),
    },
    'gs': {
        'flag': True,
        'splitter': sklearn.model_selection.TimeSeriesSplit(n_splits=3),
        'hp_grid': hp_grid,
        'verbose': 1000,
        'n_jobs': 1,
        'runs': None,
        'metrics': ['score', 'precision', 'custom'],
    },
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
