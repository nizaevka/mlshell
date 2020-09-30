import difflib
import filecmp
import glob
import importlib.util
import shutil
import sys
import time

import pandas as pd
import pycnfg
import pytest

cnfg_default = None


def file_diff(filepath1, filepath2):
    filename1 = filepath1.split('/')[-1]
    filename2 = filepath2.split('/')[-1]
    with open(filepath1, 'r') as hosts0:
        with open(filepath2, 'r') as hosts1:
            diff = difflib.unified_diff(
                hosts0.readlines(),
                hosts1.readlines(),
                fromfile=filename1,
                tofile=filename2,
                n=0,
            )
            for line in diff:
                for prefix in ('---', '+++', '@@'):
                    if line.startswith(prefix):
                        break
                else:
                    sys.stdout.write(line[1:])


def get_params(pyfile, module_name, obj_name=None):
    """Import test params."""
    conf = pyfile
    spec = importlib.util.spec_from_file_location(module_name, f"{conf}")
    conf_file = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conf_file)
    if obj_name is not None:
        obj = getattr(conf_file, obj_name)   # conf_file.params.
    else:
        obj = conf_file
    return obj


def runs_loader(path):
    """Import runs.csv as DataFrame."""
    files = sorted(glob.glob(f"{path}/*_runs.csv"))
    df_lis = list(range(len(files)))
    for i, f in enumerate(files):
        try:
            df_lis[i] = pd.read_csv(f, sep=",", header=0)
            print('Read runs.csv\n', f, df_lis[i].shape,
                  df_lis[i]['dataset__id'][0], df_lis[i]['pipeline__id'][0])
        except Exception as e:
            print(e)
            continue
    df = pd.concat(df_lis, axis=0, sort=False).reset_index()
    # with pd.option_context('display.max_rows', None,
    #                        'display.max_columns', None):
    #     msg = tabulate.tabulate(df, headers='keys', tablefmt='psql')
    #     print(msg)
    return df


@pytest.mark.parametrize("id_,args,kwargs,expected",
                         get_params(f"{__file__.replace('.py', '')}/params.py",
                                    'temp', 'params'))
def test_run(id_, args, kwargs, expected):
    """
    - Delete previous test output if exist.
    - Start mlshell.run.py.
    - Check current output with original.

    """
    # Remove results for test.
    results_path = expected['results_path']
    shutil.rmtree(results_path, ignore_errors=True)
    # Start code.
    # [future] attempts to run classification with n_jobs>1
    # global cnfg_default
    # sys.modules['cnfg_default'] = get_params(args[0], 'cnfg_default')
    # import cnfg_default
    # #from cnfg_default import custom_score_metric
    objects = pycnfg.run(oid='default', *args, **kwargs)
    tmp = {k: type(v).__name__ for k, v in objects.items()}
    print('OBJECTS:')
    print(tmp)
    # Compare results:
    # * Compare objects (keys and str of values).
    objects_ = expected['objects']
    objects = {k: type(v).__name__ for k, v in objects.items()}
    assert objects == objects_
    # for k, v in objects.items():
    #     assert k in objects_
    #     assert type(v).__name__ == objects_[k]
    # * Compare predictions csv(all available).
    pred_path = glob.glob(f"{results_path}/models/*_pred.csv")
    pred_path_ = glob.glob(expected['pred_path'])
    assert len(pred_path) == len(pred_path_)
    for act, exp in zip(sorted(pred_path), sorted(pred_path_)):
        file_diff(act, exp)
        assert filecmp.cmp(act, exp)
    # * Compare test logs.
    logs_path = glob.glob(f"{results_path}/logs*/*_test.log")[0]
    logs_path_ = expected['logs_path']
    assert filecmp.cmp(logs_path, logs_path_)
    # * Compare runs dataframe, non-universe columns.
    runs_path = f"{results_path}/runs"
    runs_path_ = expected['runs_path']
    df = runs_loader(runs_path)
    df_ = runs_loader(runs_path_)
    # First False/True for each element, then check all by columns.
    # col1     True
    # col2    False
    # dtype: bool
    df_diff = df.eq(df_).all()
    # Column names that are not equal.
    columns = sorted(list(df_diff[df_diff==False].dropna().index))
    # columns_eq = sorted(list(df_diff[df_diff==True].dropna().index))
    columns_ = expected['columns_diff']
    print('DIFF:\n', columns)
    time.sleep(1)
    # assert columns == columns_
    # * Compare model.
    model_path = glob.glob(f"{results_path}/models/*.model")
    model_path_ = glob.glob(expected['model_path'])
    assert len(model_path) == len(model_path_)
    for act, exp in zip(sorted(model_path), sorted(model_path_)):
        assert filecmp.cmp(act, exp)
    return


# Otherwise can`t pickle when n_jobs>1, need import to address scope.
# custom_score_metric = get_params('classification/conf.py',
#                                  'custom_score_metric')
# sys.modules[f'custom_score_metric'] = custom_score_metric