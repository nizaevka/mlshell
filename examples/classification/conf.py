"""Classification model tuning example.

https://www.kaggle.com/c/ieee-fraud-detection

Current configuration:
* use 10000 rows subset of train and test.
* use lgbm.
* custom metric example to pass_custom__kw_args.
* two-stage optimization:
    1. default mlshell.model_selection.RandomizedSearchOptimizer on 'roc_auc':
     'estimate__classifier__num_leaves'/ 'pass_custom__kw_args' hps.
    2. efficient mlshell.model_selection.MockOptimizer on custom metric:
     'estimate__apply_threshold__threshold', grid values (10 samples) auto
     resolved with ROC curve on predictions of first stage best estimator.

"""

import lightgbm
import mlshell
import numpy as np
import pycnfg
import sklearn
import xgboost
import pandas as pd


# Set hp ranges for optimization stage 1.
hp_grid_1 = {
    # TODO: test with Mock.
    'pass_custom__kw_args': [{'param_a': 1, 'param_b': 'c'}, {'param_a': 2, 'param_b': 'd'}, ],
    # 'process_parallel__pipeline_numeric__impute__gaps__strategy': ['median', 'constant'],
    # 'process_parallel__pipeline_numeric__transform_normal__skip': [True],
    # 'process_parallel__pipeline_numeric__scale_column_wise__quantile_range': [(0, 100), (1, 99)],
    # 'process_parallel__pipeline_numeric__add_polynomial__degree': [1],

    # # lgbm
    # 'estimate__classifier__n_estimators': np.linspace(50, 1000, 10, dtype=int),
    'estimate__predict_proba__classifier__num_leaves': [2**i for i in range(1, 5 + 1)],
    # 'estimate__classifier__min_child_samples': scipy.stats.randint(1, 100),
    # 'estimate__classifier__max_depth': np.linspace(1, 30, 10, dtype=int),
}

# Set hp ranges for optimization stage 2.
hp_grid_2 = {
    'estimate__apply_threshold__threshold': 'auto',  # Auto-resolving.
}


def custom_score_metric(y_true, y_pred, **kw_args):
    """Custom precision metric with kw_args supporting."""
    if kw_args:
        # `pass_custom_kw_args` are passed here.
        # some logic.
        print(kw_args)
    tp = np.count_nonzero((y_true == 1) & (y_pred == 1))
    fp = np.count_nonzero((y_true == 0) & (y_pred == 1))
    score = tp/(fp+tp) if tp+fp != 0 else 0
    return score


def merge(self, dataset, left_id, right_id, **kwargs):
    """Patch to DatasetProducer, add step to merge dataframe."""
    left = dataset[left_id]
    right = dataset[right_id]
    raw = pd.merge(left, right, **kwargs)
    # test dataset contains mistakes in column name.
    raw.columns = [i.replace('-', '_') for i in raw.columns]
    dataset['data'] = raw
    return dataset


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
                'th_step': True,
            }
        },
        'lgbm': {
            'kwargs': {
                'estimator_type': 'classifier',
                'estimator': lightgbm.LGBMClassifier(
                    objective='binary', n_estimators=500, num_leaves=32,
                    min_child_samples=1, max_depth=13, learning_rate=0.03,
                    boosting_type='gbdt', subsample_freq=3, subsample=0.9,
                    reg_alpha=0.3, reg_lambda=0.3, colsample_bytree=0.9,
                    silent=True, n_jobs=-1, random_state=42),
                'th_step': True,
            }
        },
        'xgb': {
            'kwargs': {
                'estimator_type': 'classifier',
                'estimator': xgboost.XGBClassifier(
                    objective='binary:hinge', **{
                        'min_child_weight': 1, 'eta': 0.01,
                        'n_estimators': 100, 'colsample_bytree': 0.5,
                        'max_depth': 12, 'subsample': 0.8, 'alpha': 1,
                        'gamma': 1, 'silent': 1, 'verbose_eval': True,
                        'seed': 42,
                    }),
                'th_step': True,
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
        'global': {'targets_names': ['isFraud'],
                   'categor_names': [
                       'ProductCD', 'addr1', 'addr2', 'P_emaildomain',
                       'R_emaildomain', 'DeviceType', 'DeviceInfo',
                       *[f'M{i}' for i in range(1, 10)],
                       *[f'card{i}' for i in range(1, 7)],
                       *[f'id_{i}' for i in range(12, 39)]
                   ],
                   'load__kwargs': {'nrows': 10000,
                                    'index_col': 'TransactionID'},
                   },
        'patch': {'merge': merge},
        'train': {
            'steps': [
                ('load', {'filepath': 'data/train_transaction.csv',
                          'key': 'transaction'}),
                ('load', {'filepath': 'data/train_identity.csv',
                          'key': 'identity'}),
                ('merge', {'left_id': 'transaction', 'right_id': 'identity',
                           'left_index': True, 'right_index': True,
                           'how': 'left', 'suffixes': ('_left', '_right')}),
                ('info',),
                ('preprocess',),
                ('split', {'train_size': 0.7, 'shuffle': False,
                           'random_state': 42}),
            ],
        },
        'test': {
            'steps': [
                ('load', {'filepath': 'data/test_transaction.csv',
                          'key': 'transaction'}),
                ('load', {'filepath': 'data/test_identity.csv',
                          'key': 'identity'}),
                ('merge', {'left_id': 'transaction', 'right_id': 'identity',
                           'left_index': True, 'right_index': True,
                           'how': 'left', 'suffixes': ('_left', '_right')}),
                ('info',),
                ('preprocess',),
            ],
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
            'metric_id': ['metric__roc_auc', 'metric__precision',
                          'metric__custom'],
            'validate__metric_id': ['metric__roc_auc', 'metric__precision',
                                    'metric__custom',
                                    'metric__classification_report',
                                    'metric__confusion_matrix'],
            'steps': [
                ('optimize', {'hp_grid': hp_grid_1,
                              'gs_params': 'gs_params__stage_1'}),
                ('optimize', {'hp_grid': hp_grid_2,
                              'gs_params': 'gs_params__stage_2',
                              'optimizer': mlshell.model_selection.MockOptimizer,
                              'resolve_params': 'resolve_params__stage_2'
                              }),
                ('validate',),
                ('predict',),
                ('dump',),
            ],

        },
    },
    # Separate section for 'resolve_params' kwarg in optimize.
    'resolve_params': {
        'stage_2': {
            'priority': 3,
            'init': {
                'estimate__apply_threshold__threshold': {
                    'cross_val_predict': {
                        'method': 'predict_proba',
                        'cv': sklearn.model_selection.TimeSeriesSplit(n_splits=3),
                        'fit_params': {},
                    },
                    'calc_th_range': {
                        'samples': 10,
                        'plot_flag': False,
                    },
                },
            }
        }
    },
    # Separate section for 'gs_params' kwarg in optimize.
    'gs_params': {
        'stage_1': {
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
        'stage_2': {
            'priority': 3,
            'init': {
                'n_iter': None,
                'n_jobs': 1,
                'refit': 'metric__custom',
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
