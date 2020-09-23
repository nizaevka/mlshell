"""Smallest."""

import lightgbm
import mlshell
import pycnfg
import sklearn


# hp ranges.
hp_grid = {
    'estimate__regressor__num_leaves': [2 ** i for i in range(1, 7)],
    # 'estimate__regressor__n_estimators': [100 * i for i in range(1, 6)],
}

CNFG = {
    'pipeline': {
        'lgbm': {
            'kwargs': {
                'estimator_type': 'regressor',
                'estimator': lightgbm.LGBMRegressor(
                    num_leaves=2, n_estimators=200,
                    max_depth=-1, random_state=42),
            }
        }
    },
    'metric': {
        'r2': {
            'score_func': sklearn.metrics.r2_score,
            'kwargs': {'greater_is_better': True},
        },
    },
    'dataset': {
        'global': {'targets_names': ['wage'],
                   'categor_names': ['union', 'goodhlth', 'black', 'female',
                                     'married', 'service'],
                   },
        'train': {
            'steps': [
                ('load', {'filepath': './data/train.csv'}),
                ('info',),
                ('preprocess',),
                ('split', {'train_size': 0.75, 'shuffle': False, }),
            ],
        },
        'test': {
            'steps': [
                ('load', {'filepath': 'data/test.csv'}),
                ('info',),
                ('preprocess', ),
            ],
        },
    },
    'workflow': {
        'conf': {
            'global': {
                'pipeline_id': 'pipeline__lgbm',
                'metric_id': ['metric__r2'],
            },
            'steps': [
                ('optimize', {
                    'dataset_id': 'dataset__train',
                    'hp_grid': hp_grid,
                    'gs_params': {
                        'n_iter': None,
                        'n_jobs': 1,
                        'refit': 'metric__r2',
                    }}),
                ('validate', {'dataset_id': 'dataset__train'}),
                ('predict', {'dataset_id': 'dataset__test'}),
                ('dump',),
            ],
        },
    },
}


if __name__ == '__main__':
    objects = pycnfg.run(CNFG, dcnfg=mlshell.CNFG)
