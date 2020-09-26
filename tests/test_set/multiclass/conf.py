"""Configuration example - tune LGBM on iris dataset.
MULTICLASS
"""
import lightgbm
import mlshell
import pycnfg
import sklearn.multioutput
import sklearn.datasets

iris_df = sklearn.datasets.load_iris(as_frame=True).frame
# [deprecated] sklearn not support multiclass-multioutput.
# iris_df.insert(0, 'target2', iris_df['target'])

# Optimization hp ranges.
hp_grid_1 = {
    'reduce_dimensions__skip': [False, True],  # PCA on/off
    # 'estimate': [
    #     lightgbm.sklearn.LGBMClassifier(
    #         num_leaves=5, max_depth=5, n_estimators=100,
    #         random_state=42),
    #     lightgbm.sklearn.LGBMRegressor(
    #         num_leaves=5, max_depth=5, n_estimators=100,
    #         random_state=42)
    # ]
    # 'estimate__classifier__n_estimators': np.linspace(50, 1000, 10, dtype=int),
    # ...
}

# Set hp ranges for optimization stage 2.
hp_grid_2 = {
    'estimate__apply_threshold__threshold': 'auto',  # Auto-resolving.
}

# Set hp ranges for optimization stage 3.
hp_grid_3 = {
    'pass_custom__kw_args': [
        {'metric__custom': {'param_a': 1, 'param_b': 'c'}},
        {'metric__custom': {'param_a': 2, 'param_b': 'd'}}
    ],
}

# Set hp ranges for optimization stage 4 (threshold and pass_custom together).
hp_grid_4 = {
    'estimate__apply_threshold__threshold': 'auto',
    'pass_custom__kw_args': [{'metric__custom': {'param_a': 1, 'param_b': 'c'}},
                             {'metric__custom': {'param_a': 2, 'param_b': 'd'}}],
}

CNFG = {
    # Pipeline section - make pipeline object(s).
    'pipeline': {
        'lgbm': {
            'init': mlshell.Pipeline,
            'producer': mlshell.PipelineProducer,
            'priority': 3,
            'steps': [
                ('make', {
                    'estimator_type': 'classifier',
                    'steps': mlshell.pipeline.Steps,
                    'estimator': lightgbm.sklearn.LGBMClassifier(
                            num_leaves=5, max_depth=5, n_estimators=100,
                            random_state=42),  # last stage of pipeline.
                    'th_step': True,
                }),
            ],
        }
    },
    # Metric section - make scorer object(s).
    'metric': {
        'accuracy': {
            'init': mlshell.Metric,
            'producer': mlshell.MetricProducer,
            'priority': 4,
            'steps': [
                ('make', {
                    'score_func': sklearn.metrics.accuracy_score,
                    'greater_is_better': True,
                }),
            ],
        },
        'confusion_matrix': {
            'init': mlshell.Metric,
            'producer': mlshell.MetricProducer,
            'priority': 4,
            'steps': [
                ('make', {
                    'score_func': sklearn.metrics.confusion_matrix,
                }),
            ],
        },
    },
    # Dataset section - dataset loading/preprocessing/splitting.
    'dataset': {
        'train': {
            'init': mlshell.Dataset({
                'data': iris_df,
            }),
            'producer': mlshell.DatasetProducer,
            'priority': 5,
            'steps': [
                ('preprocess', {'targets_names': ['target']}),
                ('split', {'train_size': 0.75, 'shuffle': True,
                           'random_state': 42}),
            ],
        },
    },
    # Workflow section
    # - fit/predict pipelines on datasets,
    # - optimize/validate metrics,
    # - predict/dump predictions on datasets.
    'workflow': {
        'conf': {
            'init': {},
            'producer': mlshell.Workflow,
            'priority': 6,
            'pipeline_id': 'pipeline__lgbm',
            'dataset_id': 'dataset__train',
            'subset_id': 'train',
            'metric_id': ['metric__accuracy'],
            'steps': [
                ('fit', {'hp': hp_grid_1}),
                ('optimize', {'hp_grid': hp_grid_1,
                              'gs_params': 'gs_params__stage_1'}),
                # Threshold.
                ('optimize', {'hp_grid': hp_grid_2,
                              'gs_params': 'gs_params__stage_2',
                              'optimizer': mlshell.model_selection.MockOptimizer,
                              'resolve_params': 'resolve_params__stage_2'
                              }),
                # Pass custom.
                ('optimize', {'hp_grid': hp_grid_3,
                              'gs_params': 'gs_params__stage_3',
                              'optimizer': mlshell.model_selection.MockOptimizer,
                              }),
                # Threshold + Pass custom.
                ('optimize', {'hp_grid': hp_grid_4,
                              'gs_params': 'gs_params__stage_2',
                              'optimizer': mlshell.model_selection.MockOptimizer,
                              'resolve_params': 'resolve_params__stage_2'
                              }),
                ('optimize', {'hp_grid': {},
                              'gs_params': 'gs_params__stage_1'}),
                ('optimize', {'hp_grid': {},
                              'optimizer': mlshell.model_selection.MockOptimizer,
                              'gs_params': 'gs_params__stage_2'}),
                ('validate',),
                ('predict',),
                ('dump',),
                ('validate', {
                    'subset_id': ['train', 'test'],
                    'metric_id': ['metric__accuracy',
                                  'metric__confusion_matrix'],
                }),
            ],
        },
    },
    'resolve_params': {
        'stage_2': {
            'priority': 3,
            'init': {
                'estimate__apply_threshold__threshold': {
                    'cross_val_predict': {
                        'method': 'predict_proba',
                        'cv': sklearn.model_selection.KFold(n_splits=3,
                                                            shuffle=True,
                                                            random_state=42),
                        'fit_params': {},
                    },
                    'calc_th_range': {
                        'samples': 10,
                        'plot_flag': False,
                        'multi_class': 'ovr',
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
                'refit': 'metric__accuracy',
                'cv': sklearn.model_selection.KFold(n_splits=3,
                                                    shuffle=True,
                                                    random_state=42),
                'verbose': 1,
                'pre_dispatch': 'n_jobs',
                'return_train_score': True,
            },
        },
        'stage_2': {
            'priority': 3,
            'init': {
                'method': 'predict_proba',
                'n_iter': None,
                'n_jobs': 1,
                'refit': 'metric__accuracy',
                'cv': sklearn.model_selection.KFold(n_splits=3,
                                                    shuffle=True,
                                                    random_state=42),
                'verbose': 1,
                'pre_dispatch': 'n_jobs',
                'return_train_score': True,
            },
        },
        'stage_3': {
            'priority': 3,
            'init': {
                'method': 'predict_proba',
                'n_iter': None,
                'n_jobs': 1,
                'refit': 'metric__accuracy',
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
    # mlshell.CNFG contains default section / configuration keys for typical ml
    # task, including pretty logger and project path detection.
    objects = pycnfg.run(CNFG, dcnfg=mlshell.CNFG, update_expl=False)
