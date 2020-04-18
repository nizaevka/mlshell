"""The module to set user configuration parammeters.

Attributes:
    params: user parameters
        see mlshell.default module for detailed description of parameters

"""

import numpy as np
import sklearn
import lightgbm
import xgboost


# choose estimator
estimator = [
    # sklearn.linear_model.SGDClassifier(loss='log',
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
    lightgbm.LGBMClassifier(objective='binary', num_leaves=2, min_child_samples=1,
                            n_estimators=250, max_depth=-1, silent=False)
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
    #     })
    ][0]


# define hyperparameters (hp) to cv
# set ranges for hp
hp_grid = {
    # 'process_parallel__pipeline_numeric__impute__gaps__strategy': ['median', 'constant'],
    'process_parallel__pipeline_numeric__transform_normal__skip': [True],
    # 'process_parallel__pipeline_numeric__scale_column_wise__quantile_range': [(0, 100), (1, 99)],
    'process_parallel__pipeline_numeric__add_polynomial__degree': [1],

    # # lgbm
    # 'estimate__classifier__n_estimators': np.linspace(50, 1000, 10, dtype=int),
    # 'estimate__classifier__num_leaves': [2**i for i in range(1, 2 + 1)],
    # 'estimate__classifier__min_data_in_leaf': np.linspace(10, 100, 10, dtype=int),
    # 'estimate__classifier__max_depth': np.linspace(1, 30, 10, dtype=int),
    # 'estimate__apply_threshold__threshold': [0.8]

    # LC
    # 'estimate__classifier__alpha': np.logspace(-4, 1, num=5),

    # XGBOOST
    'estimate__classifier__n_estimators': np.linspace(50, 100, 2, dtype=int),
}


# classifier custom metric example
def custom_score_metric(y_true, y_pred):
    # metrics.confusion_matrix(y_true, y_pred)
    tp = np.count_nonzero((y_true == 1) & (y_pred == 1))
    fp = np.count_nonzero((y_true == 0) & (y_pred == 1))
    # tp_fn = np.count_nonzero(y_true == 1)
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
        'confusion matrix': (sklearn.metrics.confusion_matrix, {'labels': [0, 1]}),
        'classification report': (sklearn.metrics.classification_report, {'output_dict': True, 'zero_division': 0}),
    },
    'gs': {
        'flag': True,
        'splitter': sklearn.model_selection.KFold(n_splits=3, shuffle=True),
        'hp_grid': hp_grid,
        'verbose': 1000,
        'n_jobs': 1,
        'runs': None,
        'metrics': ['score', 'precision', 'custom'],
    },
    'data': {
        'train': {
            'args': ['data/train_transaction_5k.csv',
                     'data/train_identity_5k.csv'],
            'kw_args': {'rows_limit': 2000,
                        'random_skip': False,
                        'index_col': 'TransactionID'},
        },
        'test': {
            'args': ['data/test_transaction_5k.csv',
                     'data/test_identity_5k.csv'],
            'kw_args': {'rows_limit': 2000,
                        'random_skip': False,
                        'index_col': 'TransactionID'},
        },
        'split_train_size': 0.7,
        'del_duplicates': False,
    },
    'th': {
        'pos_label': 1,
        'strategy': 1,
        'samples': 10,
        'plot_flag': False,
    },
    'cache': {
        'pipeline': False,
        'unifier': False,
    },
    'seed': 42,
}
