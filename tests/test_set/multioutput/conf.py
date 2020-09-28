"""Configuration example.
binary iris dataset.
MULTIOUTPUT
* Regression
    - ** from the box(no such estimators in sklearn)
    ** MultiOutputRegressor
* Classification
    ** from the box with th_step
    ** MultiOutputClassifierx with th_step
    - ** multiclass-multioutput (RandomForestClassifier, no such metrics in sklearn)

"""
import lightgbm
import mlshell
import pycnfg
import sklearn.multioutput
import sklearn.datasets
import sklearn.ensemble


iris_df = sklearn.datasets.load_iris(as_frame=True).frame
# Cast to binary (rare support multiclass-multioutput).
# Multioutput
iris_df.insert(0, 'target2', iris_df['target'])
# binary
iris_bin_df = iris_df.copy(deep=True)
iris_bin_df.loc[(iris_df.target > 1), 'target'] = 1
iris_bin_df.loc[(iris_df.target2 > 1), 'target2'] = 1

# There is no multioutput regression from the box.
regressor = lightgbm.sklearn.LGBMRegressor(
    num_leaves=5, max_depth=5, n_estimators=100, random_state=42)
classifier_box = sklearn.ensemble.RandomForestClassifier(random_state=42)  # support multiclass-multioutput
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
            'estimator': sklearn.multioutput.MultiOutputClassifier(classifier),
            'kwargs': {'th_step': True},
        },
        '2': {
            'estimator': classifier_box,
            'kwargs': {'th_step': True},
        },
        '3': {
            'estimator': sklearn.multioutput.MultiOutputRegressor(regressor)
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
        'r2': {
            'score_func': sklearn.metrics.r2_score,
            'kwargs': {'greater_is_better': True},
        },
    },
    # Dataset section - dataset loading/preprocessing/splitting.
    'dataset': {
        'binary': {
            'init': mlshell.Dataset({
                'data': iris_bin_df,
            }),
            'producer': mlshell.DatasetProducer,
            'priority': 5,
            'steps': [
                ('preprocess', {'targets_names': ['target', 'target2']}),
                ('split', {'train_size': 0.75, 'shuffle': True,
                           'random_state': 42}),
            ],
        },
        'multiclass': {
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
            'dataset_id': 'dataset__binary',
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
            }),
        ],
        # classification MultiOutput(binary)
        '1': {
            'priority': 6,
            'pipeline_id': 'pipeline__1',
        },
        # classification from the box (multioutput)
        '2': {
            'priority': 6,
            # 'dataset_id': 'dataset__multiclass', (no such metrics)
            'pipeline_id': 'pipeline__2',
        },
        # regression MultiOutput
        '3': {
            'priority': 6,
            'pipeline_id': 'pipeline__3',
            'metric_id': ['metric__r2'],
            'gs_params': 'gs_params__stage_4',
            'steps': [
                ('fit', {'hp': hp_grid_1}),
                ('optimize', {'hp_grid': hp_grid_1}),
                # Pass custom.
                ('optimize', {'hp_grid': hp_grid_3,
                              'optimizer': mlshell.model_selection.MockOptimizer,
                              }),
                ('optimize', {'hp_grid': {}}),
                ('optimize', {'hp_grid': {},
                              'optimizer': mlshell.model_selection.MockOptimizer,
                              }),
                ('validate',),
                ('predict',),
                ('dump',),
                ('validate', {
                    'subset_id': ['train', 'test'],
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
                        # 'multi_output': 'product',
                    },
                },
            }
        }
    },
    # Separate section for 'gs_params' kwarg in optimize.
    'gs_params': {
        # classification
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
        # regression
        'stage_4': {
            'priority': 3,
            'init': {
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
        },
    },
}


if __name__ == '__main__':
    # mlshell.CNFG contains default section / configuration keys for typical ml
    # task, including pretty logger and project path detection.
    objects = pycnfg.run(CNFG, dcnfg=mlshell.CNFG)
