"""Configuration example.
binary iris dataset.
MULTIOUTPUT
* Regression
    ** from the box
    ** MultiOutputRegressor
* Classification
    ** from the box with th_step
    ** MultiOutputClassifier
"""
import lightgbm
import mlshell
import pycnfg
import sklearn.multioutput
import sklearn.datasets


iris_df = sklearn.datasets.load_iris(as_frame=True).frame
# Cast to binary sklearn not support multiclass-multioutput.
iris_df.loc[(iris_df.target > 1), 'target'] = 1
# Multioutput
iris_df.insert(0, 'target2', iris_df['target'])

regressor_box = sklearn.linear_model.SGDRegressor()
regressor = lightgbm.sklearn.LGBMRegressor(
    num_leaves=5, max_depth=5, n_estimators=100, random_state=42)
classifier_box = sklearn.linear_model.SGDClassifier()
classifier = lightgbm.sklearn.LGBMClassifier(
    num_leaves=5, max_depth=5, n_estimators=100, random_state=42)

# Optimization hp ranges.
hp_grid_1 = {
    'reduce_dimensions__skip': [True],  # PCA on/off
    # 'estimate': [
    #     sklearn.multioutput.MultiOutputClassifier(classifier),
    #     sklearn.multioutput.MultiOutputRegressor(regressor),
    #     sklearn.compose.TransformedTargetRegressor(regressor=regressor_box),
    #     sklearn.pipeline.Pipeline(steps=[
    #         ('predict_proba',
    #             mlshell.model_selection.PredictionTransformer(classifier_box)),
    #         ('apply_threshold',
    #             mlshell.model_selection.ThresholdClassifier(
    #                 params='auto', threshold=None)),
    #         ]),
    # ],
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
        '1': {
            'estimator': sklearn.multioutput.MultiOutputClassifier(classifier)
        },
        '2': {
            'estimator': sklearn.multioutput.MultiOutputRegressor(regressor)
        },
        '3': {
            'estimator': sklearn.compose.TransformedTargetRegressor(regressor=regressor_box)
        },
        '4': {
            'estimator': sklearn.pipeline.Pipeline(steps=[
                ('predict_proba',
                    mlshell.model_selection.PredictionTransformer(classifier_box)),
                ('apply_threshold',
                    mlshell.model_selection.ThresholdClassifier(
                        params='auto', threshold=None)),
                ])
        },

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
                ('preprocess', {'targets_names': ['target', 'target2']}),
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
        'global': {
            'dataset_id': 'dataset__train',
            'subset_id': 'train',
            'metric_id': ['metric__accuracy'],
        },
        'priority': 6,
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
        '1': {
            'pipeline_id': 'pipeline__1',
        },
        '2': {
            'pipeline_id': 'pipeline__2',
        },
        '3': {
            'pipeline_id': 'pipeline__3',
        },
        '4': {
            'pipeline_id': 'pipeline__4',
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
