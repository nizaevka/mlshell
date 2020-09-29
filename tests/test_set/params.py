"""
Params to test mlshell based configuration in `pycnfg.run`.
[(args, kwargs, result), ..].
Only for import (relate on workdir).
"""
import pathlib
import platform

import mlshell

currdir = pathlib.Path(__file__).parent.absolute()
workdir = pathlib.Path().absolute()
os_type = platform.system()
output = {
    'objects': {},
    'columns_diff': [],  # Columns diff because off func address diffs.
    'results_path': f"{currdir}/__id__/results",
    'logs_path': f"{currdir}/__id__/original/logs/test_{os_type}.log",
    'pred_path': f"{currdir}/__id__/original/models/*{os_type}*_pred.csv",
    'runs_path': f"{currdir}/__id__/original/runs",
    'model_path': f"{currdir}/__id__/original/models/*{os_type}*.model",
}


def sbst_id(dic, id_, upd=None):
    """Substitute test id and upd dict.

    Parameters
    ----------
    dic : dict
    id_ : str
        Subdir name with specific test results.
    upd : dict
         Update for dic.
    """
    if upd is None:
        upd = {}
    return {**{k: (v.replace('__id__', id_) if isinstance(v, str) else v)
            for k, v in dic.items()}, **upd}


# out = {'logger__default': logger, 'path__default': str(workdir)}

params = [
    # Regression.
    (
        0,
        [f"{currdir}/regression/conf.py"],
        {'dcnfg': mlshell.CNFG,
         'objects': {'path__default': f"{currdir}/regression/"}},
        sbst_id(output, 'regression', upd={
            # merged runs.csv columns that are differ between original/current.
            'columns_diff': sorted([
                'steps', 'pass_custom', 'select_rows',
                'process_parallel', 'pass_custom__func', 'select_rows__func',
                'process_parallel__transformer_list',
                'process_parallel__pipeline_categoric',
                'process_parallel__pipeline_numeric',
                'process_parallel__pipeline_categoric__steps',
                'process_parallel__pipeline_categoric__select_columns',
                'process_parallel__pipeline_categoric__select_columns__func',
                'process_parallel__pipeline_numeric__steps',
                'process_parallel__pipeline_numeric__select_columns',
                'process_parallel__pipeline_numeric__compose_columns',
                'process_parallel__pipeline_numeric__select_columns__func',
                'process_parallel__pipeline_numeric__impute__indicators__missing_values',
                'process_parallel__pipeline_numeric__impute__gaps__missing_values',
                'process_parallel__pipeline_numeric__compose_columns__transformers',
                'estimate__transformer__accept_sparse',
                'estimate__transformer__check_inverse',
                'estimate__transformer__func',
                'estimate__transformer__inv_kw_args',
                'estimate__transformer__inverse_func',
                'estimate__transformer__kw_args',
                'estimate__transformer__validate',
                'mean_fit_time', 'std_fit_time', 'mean_score_time',
                'std_score_time', 'id', 'pipeline__hash'
            ]),
            # print of objects result.
            'objects': {'path__default': 'str',
                        'logger__default': 'LoggerAdapter',
                        'dataset__test': 'Dataset',
                        'dataset__train': 'Dataset',
                        'gs_params__conf_1': 'dict',
                        'metric__mae': 'Metric',
                        'metric__r2': 'Metric',
                        'pipeline__lgbm': 'Pipeline',
                        'pipeline__sgd': 'Pipeline',
                        'workflow__conf': 'dict'}
        }),
    ),
    # Classification.
    (
        1,
        [f"{currdir}/classification/conf.py"],
        {'dcnfg': mlshell.CNFG,
         'objects': {'path__default': f"{currdir}/classification/"}},
        sbst_id(output, 'classification', upd={
            'columns_diff': sorted([
                'steps',
                'pass_custom',
                'select_rows',
                'process_parallel', 'pass_custom__func',
                'select_rows__func', 'process_parallel__transformer_list',
                'process_parallel__pipeline_categoric',
                'process_parallel__pipeline_numeric',
                'process_parallel__pipeline_categoric__steps',
                'process_parallel__pipeline_categoric__select_columns',
                'process_parallel__pipeline_categoric__select_columns__func',
                'process_parallel__pipeline_numeric__steps',
                'process_parallel__pipeline_numeric__select_columns',
                'process_parallel__pipeline_numeric__compose_columns',
                'process_parallel__pipeline_numeric__select_columns__func',
                'process_parallel__pipeline_numeric__impute__indicators__missing_values',
                'process_parallel__pipeline_numeric__impute__gaps__missing_values',
                'process_parallel__pipeline_numeric__compose_columns__transformers',
                'mean_fit_time',
                'std_fit_time',
                'mean_score_time',
                'std_score_time',
                'id', 'pipeline__hash'
            ]),
            'objects': {'path__default': 'str',
                        'logger__default': 'LoggerAdapter',
                        'dataset__test': 'Dataset',
                        'dataset__train': 'Dataset',
                        'gs_params__stage_1': 'dict',
                        'gs_params__stage_2': 'dict',
                        'gs_params__stage_3': 'dict',
                        'metric__classification_report': 'Metric',
                        'metric__confusion_matrix': 'Metric',
                        'metric__custom': 'Metric',
                        'metric__precision': 'Metric',
                        'metric__roc_auc': 'Metric',
                        'pipeline__lgbm': 'Pipeline',
                        'pipeline__sgd': 'Pipeline',
                        'resolve_params__stage_2': 'dict',
                        'workflow__conf': 'dict'}
        }),
    ),
    # Whole estimator (whithout steps).
    (
        2,
        [f"{currdir}/whole/conf.py"],
        {'dcnfg': mlshell.CNFG,
         'objects': {'path__default': f"{currdir}/whole/"}},
        sbst_id(output, 'whole', upd={
            'columns_diff': sorted([
                'mean_fit_time',
                'std_fit_time',
                'mean_score_time',
                'std_score_time',
                'id',
                'pipeline__hash'
            ]),
            'objects': {
                'path__default': 'str',
                'logger__default': 'LoggerAdapter',
                'dataset__test': 'Dataset',
                'dataset__train': 'Dataset',
                'gs_params__conf_1': 'dict',
                'metric__mse': 'Metric',
                'pipeline__sgd': 'Pipeline',
                'workflow__conf': 'dict'}
        }),
    ),
    # multiclass.
    (
        3,
        [f"{currdir}/multiclass/conf.py"],
        {'dcnfg': mlshell.CNFG,
         'objects': {'path__default': f"{currdir}/multiclass/"}},
        sbst_id(output, 'multiclass', upd={
            'columns_diff': sorted([
                'steps',
                'pass_custom',
                'select_rows',
                'process_parallel', 'pass_custom__func',
                'select_rows__func', 'process_parallel__transformer_list',
                'process_parallel__pipeline_categoric',
                'process_parallel__pipeline_numeric',
                'process_parallel__pipeline_categoric__steps',
                'process_parallel__pipeline_categoric__select_columns',
                'process_parallel__pipeline_categoric__select_columns__func',
                'process_parallel__pipeline_numeric__steps',
                'process_parallel__pipeline_numeric__select_columns',
                'process_parallel__pipeline_numeric__compose_columns',
                'process_parallel__pipeline_numeric__select_columns__func',
                'process_parallel__pipeline_numeric__impute__indicators__missing_values',
                'process_parallel__pipeline_numeric__impute__gaps__missing_values',
                'process_parallel__pipeline_numeric__compose_columns__transformers',
                'mean_fit_time',
                'std_fit_time',
                'mean_score_time',
                'std_score_time',
                'id', 'pipeline__hash'
            ]),
            'objects': {
                'dataset__train': 'Dataset',
                'gs_params__stage_1': 'dict',
                'gs_params__stage_2': 'dict',
                'gs_params__stage_3': 'dict',
                'logger__default': 'LoggerAdapter',
                'metric__accuracy': 'Metric',
                'metric__confusion_matrix': 'Metric',
                'path__default': 'str',
                'pipeline__lgbm': 'Pipeline',
                'resolve_params__stage_2': 'dict',
                'workflow__conf': 'dict'}
        }),
    ),
    # binary/regression multioutput (no multiclass-multioutout metrics).
    (
        4,
        [f"{currdir}/multioutput/conf.py"],
        {'dcnfg': mlshell.CNFG,
         'objects': {'path__default': f"{currdir}/multioutput/"}},
        sbst_id(output, 'multioutput', upd={
            'columns_diff': sorted([
                'estimate__apply_threshold',
                'estimate__apply_threshold__params',
                'estimate__apply_threshold__threshold',
                'estimate__check_inverse',
                'estimate__func',
                'estimate__inverse_func',
                'estimate__memory',
                'estimate__predict_proba',
                'estimate__predict_proba__classifier',
                'estimate__predict_proba__classifier__bootstrap',
                'estimate__predict_proba__classifier__ccp_alpha',
                'estimate__predict_proba__classifier__class_weight',
                'estimate__predict_proba__classifier__criterion',
                'estimate__predict_proba__classifier__estimator',
                'estimate__predict_proba__classifier__estimator__boosting_type',
                'estimate__predict_proba__classifier__estimator__class_weight',
                'estimate__predict_proba__classifier__estimator__colsample_bytree',
                'estimate__predict_proba__classifier__estimator__importance_type',
                'estimate__predict_proba__classifier__estimator__learning_rate',
                'estimate__predict_proba__classifier__estimator__max_depth',
                'estimate__predict_proba__classifier__estimator__min_child_samples',
                'estimate__predict_proba__classifier__estimator__min_child_weight',
                'estimate__predict_proba__classifier__estimator__min_split_gain',
                'estimate__predict_proba__classifier__estimator__n_estimators',
                'estimate__predict_proba__classifier__estimator__n_jobs',
                'estimate__predict_proba__classifier__estimator__num_leaves',
                'estimate__predict_proba__classifier__estimator__objective',
                'estimate__predict_proba__classifier__estimator__random_state',
                'estimate__predict_proba__classifier__estimator__reg_alpha',
                'estimate__predict_proba__classifier__estimator__reg_lambda',
                'estimate__predict_proba__classifier__estimator__silent',
                'estimate__predict_proba__classifier__estimator__subsample',
                'estimate__predict_proba__classifier__estimator__subsample_for_bin',
                'estimate__predict_proba__classifier__estimator__subsample_freq',
                'estimate__predict_proba__classifier__max_depth',
                'estimate__predict_proba__classifier__max_features',
                'estimate__predict_proba__classifier__max_leaf_nodes',
                'estimate__predict_proba__classifier__max_samples',
                'estimate__predict_proba__classifier__min_impurity_decrease',
                'estimate__predict_proba__classifier__min_impurity_split',
                'estimate__predict_proba__classifier__min_samples_leaf',
                'estimate__predict_proba__classifier__min_samples_split',
                'estimate__predict_proba__classifier__min_weight_fraction_leaf',
                'estimate__predict_proba__classifier__n_estimators',
                'estimate__predict_proba__classifier__n_jobs',
                'estimate__predict_proba__classifier__oob_score',
                'estimate__predict_proba__classifier__random_state',
                'estimate__predict_proba__classifier__verbose',
                'estimate__predict_proba__classifier__warm_start',
                'estimate__regressor',
                'estimate__regressor__estimator',
                'estimate__regressor__estimator__boosting_type',
                'estimate__regressor__estimator__class_weight',
                'estimate__regressor__estimator__colsample_bytree',
                'estimate__regressor__estimator__importance_type',
                'estimate__regressor__estimator__learning_rate',
                'estimate__regressor__estimator__max_depth',
                'estimate__regressor__estimator__min_child_samples',
                'estimate__regressor__estimator__min_child_weight',
                'estimate__regressor__estimator__min_split_gain',
                'estimate__regressor__estimator__n_estimators',
                'estimate__regressor__estimator__n_jobs',
                'estimate__regressor__estimator__num_leaves',
                'estimate__regressor__estimator__objective',
                'estimate__regressor__estimator__random_state',
                'estimate__regressor__estimator__reg_alpha',
                'estimate__regressor__estimator__reg_lambda',
                'estimate__regressor__estimator__silent',
                'estimate__regressor__estimator__subsample',
                'estimate__regressor__estimator__subsample_for_bin',
                'estimate__regressor__estimator__subsample_freq',
                'estimate__regressor__n_jobs',
                'estimate__steps',
                'estimate__transformer',
                'estimate__verbose',
                'id',
                'mean_fit_time',
                'mean_score_time',
                'mean_test_metric__accuracy',
                'mean_test_metric__r2',
                'mean_train_metric__accuracy',
                'mean_train_metric__r2',
                'pass_custom',
                'pass_custom__func',
                'pipeline__hash',
                'process_parallel',
                'process_parallel__pipeline_categoric',
                'process_parallel__pipeline_categoric__select_columns',
                'process_parallel__pipeline_categoric__select_columns__func',
                'process_parallel__pipeline_categoric__steps',
                'process_parallel__pipeline_numeric',
                'process_parallel__pipeline_numeric__compose_columns',
                'process_parallel__pipeline_numeric__compose_columns__transformers',
                'process_parallel__pipeline_numeric__impute__gaps__missing_values',
                'process_parallel__pipeline_numeric__impute__indicators__missing_values',
                'process_parallel__pipeline_numeric__select_columns',
                'process_parallel__pipeline_numeric__select_columns__func',
                'process_parallel__pipeline_numeric__steps',
                'process_parallel__transformer_list',
                'rank_test_metric__accuracy',
                'rank_test_metric__r2',
                'select_rows',
                'select_rows__func',
                'split0_test_metric__accuracy',
                'split0_test_metric__r2',
                'split0_train_metric__accuracy',
                'split0_train_metric__r2',
                'split1_test_metric__accuracy',
                'split1_test_metric__r2',
                'split1_train_metric__accuracy',
                'split1_train_metric__r2',
                'split2_test_metric__accuracy',
                'split2_test_metric__r2',
                'split2_train_metric__accuracy',
                'split2_train_metric__r2',
                'std_fit_time',
                'std_score_time',
                'std_test_metric__accuracy',
                'std_test_metric__r2',
                'std_train_metric__accuracy',
                'std_train_metric__r2',
                'steps']),
            'objects': {
                'dataset__binary': 'Dataset',
                'dataset__multiclass': 'Dataset',
                'gs_params__stage_1': 'dict',
                'gs_params__stage_2': 'dict',
                'gs_params__stage_3': 'dict',
                'gs_params__stage_4': 'dict',
                'logger__default': 'LoggerAdapter',
                'metric__accuracy': 'Metric',
                'metric__r2': 'Metric',
                'path__default': 'str',
                'pipeline__1': 'Pipeline',
                'pipeline__2': 'Pipeline',
                'pipeline__3': 'Pipeline',
                'resolve_params__stage_2': 'dict',
                'workflow__1': 'dict',
                'workflow__2': 'dict',
                'workflow__3': 'dict'
            }
        }),
    ),
]
