"""Configuration example.

Create pipeline (sgd) and optimize hp_grid:
* target transformer on/off.
* polynomial degree 1/2.

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
    # Pipeline section - specify pipelines creation/loading.
    'pipeline': {
        'sgd': {
            'steps': [
                ('make', {
                    'estimator_type': 'regressor',
                    'estimator': sklearn.linear_model.SGDRegressor(
                        penalty='elasticnet', l1_ratio=1, shuffle=False,
                        max_iter=1000, alpha=0.02, random_state=42),
                }),
            ],
        },
        'lgbm': {
            'steps': [
                ('make', {
                    'estimator_type': 'regressor',
                    'estimator': lightgbm.LGBMRegressor(
                        num_leaves=2, min_data_in_leaf=60,
                        n_estimators=200, max_depth=-1, random_state=42),
                }),
            ],
        }
    },
    # Metric section - specify metric creation/loading.
    'metric': {
        'r2': {
            'steps': [
                ('make', {
                    'score_func': sklearn.metrics.r2_score,
                    'greater_is_better': True,
                }),
            ],
        },
        'mse': {
            'steps': [
                ('make', {
                    'score_func': sklearn.metrics.mean_squared_error,
                    'greater_is_better': True,
                    'squared': False
                }),
            ],
        },
    },
    # Dataset section - specify dataset loading/preprocessing/splitting.
    'dataset': {
        'train': {
            'steps': [
                ('load', {'filepath': './data/train.csv'}),
                ('info',),
                ('preprocess', {'targets_names': ['wage'],
                                'categor_names': ['union', 'goodhlth',
                                                  'black', 'female',
                                                  'married', 'service']}),
                ('split', {'train_size': 0.75, 'shuffle': False, }),
            ],
        },
        'test': {
            'steps': [
                ('load', {'filepath': 'data/test.csv'}),
                ('info',),
                ('preprocess', {'categor_names': ['union', 'goodhlth',
                                                  'black', 'female',
                                                  'married', 'service'],
                                'targets_names': ['wage']}),
            ],
        },
    },
    # Workflow section - fit/predict pipelines on datasets, optimize/validate
    # metrics.
    'workflow': {
        'conf': {
            'steps': [
                # Train 'lgbm' pipeline on 'train' subset of 'train' dataset
                # with zero position hp from 'hp_grid'.
                ('fit', {
                    'pipeline_id': 'pipeline__sgd',
                    'dataset_id': 'dataset__train',
                    'subset_id': 'train',
                    'hp': hp_grid,
                }),
                # Optimize 'lgbm' pipeline on 'train' subset of 'train' dataset
                # on hp combinations from 'hp_grid'. Score and refit on 'r2'
                # scorer.
                ('optimize', {
                    'pipeline_id': 'pipeline__sgd',
                    'dataset_id': 'dataset__train',
                    'subset_id': 'train',
                    'metric_id': ['metric__r2', 'metric__mse'],
                    'hp_grid': hp_grid,
                    'gs_params': {
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
                }),
                # Validate 'lgbm' pipeline on 'train' and 'test' subsets of
                # 'train' dataset with 'r2' scorer.
                ('validate', {
                    'pipeline_id': 'pipeline__sgd',
                    'dataset_id': 'dataset__train',
                    'subset_id': ['train', 'test'],
                    'metric_id': ['metric__r2', 'metric__mse'],
                }),
                # Predict with 'lgbm' pipeline on whole 'test' dataset.
                ('predict', {
                    'pipeline_id': 'pipeline__sgd',
                    'dataset_id': 'dataset__test',
                    'subset_id': '',
                }),
                # Dump 'lgbm' pipeline on disk.
                ('dump', {'pipeline_id': 'pipeline__sgd',
                          'dirpath': None}),
            ],
        },
    },
}


if __name__ == '__main__':
    pycnfg.run(CNFG, dcnfg=mlshell.CNFG)
