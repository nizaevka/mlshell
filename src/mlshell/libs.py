"""Module for library imports"""


# standard library
import cmath
import gc
import _pickle as pickle
import inspect
import shutil
import copy
import sys
import os
import json
import tempfile
import random as rd
import uuid
from hashlib import md5
import warnings
import time
import atexit


# third-party module or package
import joblib
import numpy as np
import pandas as pd
import numba as nb
import jsbeautifier
import seaborn as sns
import tabulate
from IPython import get_ipython
import lightgbm
# devs
import line_profiler
from memory_profiler import profile as memory_profiler


# code repository sub-package
import scipy.stats
import matplotlib.pyplot as plt
import sklearn.base
import sklearn.compose
import sklearn.decomposition
import sklearn.impute
import sklearn.preprocessing
import sklearn.feature_selection
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
import statsmodels.graphics.regressionplots as smg
import statsmodels.stats.outliers_influence


try:
    import winsound
except ImportError:
    def playsound(frequency, duration):
        # apt-get install beep
        os.system('beep -f %s -l %s' % (frequency, duration))
else:
    def playsound(frequency, duration):
        winsound.Beep(frequency, duration)


# TODO: raise Exception or signal when memory-profiler exceed
# https://pypi.org/project/memory-profiler/
# don`t accept multiple streams, possible variants:
#   default write in sys.stdout, redirect to file когда запускаешь скрипт
#   set explicitly one stream at every @profile  self.logger.handlers[i].stream or custom_stream
#       (при создании логера привязываем его к кастом_стрим)
#   redefine sys.stdout with custom stream (class with .write method)
#       like in built-in LogFile  sys.stdout = self.logger.handlers[i].stream

def libs_conf():
    # TODO: better use global pprofile.exe for whole script
    # create_decorator, don`t work with other decorators
    time_profiler = line_profiler.LineProfiler()
    # run print_stats at the end of script
    atexit.register(time_profiler.print_stats, output_unit=1)
    atexit.register(playsound, 600, 500)
    atexit.register(playsound, 400, 2000)  # will be the first
    # turn off FutureWarnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    # turn on: inf as NaN
    pd.options.mode.use_inf_as_na = True
    np.seterr(all='call')
    rd.seed(42)


class MyException(Exception):
    def __init__(self, msg, type='break'):
        self.msg = msg
        self.type = type


def np_divide(a, b):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        c[~np.isfinite(c)] = 0  # -inf inf NaN
    return c


if __name__ == 'main':
    pass
