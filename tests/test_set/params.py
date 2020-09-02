"""
Params to test mlshell based configuration in `pycnfg.run`.
[(args, kwargs, result), ..].
Only for import (relate on workdir).
"""
import logging
import pathlib
import sys
import mlshell
import pycnfg
import platform

currdir = pathlib.Path(__file__).parent.absolute()
workdir = pathlib.Path().absolute()
# Find out platform type.
if platform.system() == 'Windows':
    os_type = 'windows'
else:
    os_type = 'unix'

output = {
    'objects': {},
    'columns_diff': [],  # Columns diff because off func address diffs.
    'results_path': f"{currdir}/__id__/results",
    'logs_path': f"{currdir}/__id__/original/logs/test_1k_{os_type}.log",
    'pred_path': f"{currdir}/__id__/original/models/{os_type}_pred.csv",
    'runs_path': f"{currdir}/__id__/original/runs",
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
            'columns_diff': [
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
            ],
            'objects': {'path__default': 'str',
                        'logger__default': 'LoggerAdapter',
                        'dataset__test': 'Dataset',
                        'dataset__train': 'Dataset',
                        'gs_params__conf': 'dict',
                        'metric__mae': 'Metric',
                        'metric__r2': 'Metric',
                        'pipeline__lgbm': 'Pipeline',
                        'pipeline__sgd': 'Pipeline',
                        'pipeline__xgb': 'Pipeline',
                        'workflow__conf': 'dict'}
        }),
    ),
#    # Classification.
#    (
#        1,
#        [f'{currdir}/classification/conf.py'],
#        {'dcnfg': mlshell.CNFG},
#        sbst_id(output, 'classification'),
#    ),
]

print(params[0][3])