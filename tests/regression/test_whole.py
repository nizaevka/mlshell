"""The module to test code on real data

TODO
    explore: filecmp True despite of different file separator in logs. (maybe None was not extst in linux)
    explore: can`t import classes
    explore: there are difference in model and predictions for windows/unix why?
        maybe csv problem https://stackoverflow.com/questions/12877189/float64-with-pandas-to-csv
        or models difference

"""


import shutil
import runpy
import os
import glob
from . classes import GetData, DataPreprocessor
import filecmp
import platform
import logging

def same_folders(dcmp):
    """Check that two folders are copy"""
    if dcmp.diff_files:
        return False
    for sub_dcmp in dcmp.subdirs.values():
        return same_folders(sub_dcmp)
    return True


def test_func():
    """Test on readl data.

    - Delete previous test output if exist.
    - Start run.py.
    - Check current output with the one from `original` dir.

    """
    try:
        dir_path = '/'.join(__file__.replace('\\', '/').split('/')[:-1])
        shutil.rmtree(f'{dir_path}/results', ignore_errors=True)
        runpy.run_path(f'{dir_path}/run.py',
                       init_globals={'GetData': GetData, 'DataPreprocessor': DataPreprocessor},
                       run_name='regression')
        # find out platform type
        if platform.system() == 'Windows':
            os_type = 'windows'
        else:
            os_type = 'unix'

        # check GS
        assert filecmp.cmp(f'{dir_path}/original/None_critical_1k_{os_type}.log',
                           glob.glob(f'{dir_path}/results/logs_run/*_critical.log')[0])
        # check prediction
        for filepath in glob.glob(f'{dir_path}/results/models/*predictions.csv'):
            for filepath_ in glob.glob(f'{dir_path}/original/models/*predictions_{os_type}.csv'):
                assert filecmp.cmp(filepath, filepath_)

        # [deprecated] now all output in results
        # create .temp if no exist and move toutput there
        # logging.shutdown()  # otherwise error, cause  logger not close in classes
        # dest = f'{dir_path}/.temp'
        # if not os.path.exists(dest):
        #     os.makedirs(dest)
        # for dir_name in ['logs_run', 'models', 'runs']:
        #     shutil.move(f'{dir_path}/{dir_name}', dest)
    except Exception as e:
        print(e)
        assert False
