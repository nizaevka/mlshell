"""The module to set user configuration parammeters.

Attributes:
    params: user parameters.
        see mlshell.default module for detailed description of parameters.
"""

import numpy as np
import sklearn
import lightgbm
import mlshell


# choose estimator
estimator = [
    sklearn.linear_model.SGDRegressor(penalty='elasticnet', l1_ratio=1, shuffle=False, max_iter=1000, alpha=0.02)
    # lightgbm.LGBMRegressor(num_leaves=2, min_data_in_leaf=60, n_estimators=200, max_depth=-1),
    ][0]


target_transformer = sklearn.preprocessing.PowerTransformer(method='yeo-johnson', standardize=True, copy=True)


# set ranges for hp
hp_grid = {

    #'process_parallel__pipeline_numeric__transform_normal__skip': [False],
    #'process_parallel__pipeline_numeric__scale_column_wise__quantile_range': [(1, 99)],
    #'process_parallel__pipeline_numeric__add_polynomial__degree': [1],
    'estimate__transformer': [target_transformer],

    # sgd
    #'estimate__regressor__alpha': np.logspace(-2, -1, 10),
    #'estimate__regressor__l1_ratio': np.linspace(0.1, 1, 10),
}


# set workflow params
params = {
    'workflow': {
        'endpoint_id': ['endpoint_1', 'endpoint_2']
    },
    'endpoint': {
        'endpoint_1': {'global': {'gs_params': 'my_gs', 'pipeline': 'pipeline_1'}},
        'endpoint_2': {'global': {'gs_params': 'my_gs_2', 'pipeline': 'pipeline_2'}},
    },
    'pipeline': {
        'pipeline_1': {'estimator': estimator, 'type': 'regressor'},
        'pipeline_2': {'filepath': 'some', 'type': 'classifier'},
    },
    'metric': {
        'score': (sklearn.metrics.r2_score, {'greater_is_better': True}),
        'mae': (sklearn.metrics.mean_absolute_error, {'greater_is_better': False}),
        'mse': (sklearn.metrics.mean_squared_error, {'greater_is_better': False, 'squared': False}),
    },
    'gs_params': {
        'my_gs': {
            'flag': True,
            'hp_grid': hp_grid,
            'n_iter': None,  # 'gs__runs'
            'scoring': ['score', 'mae', 'mse'],
            'n_jobs': 1,
            'refit': 'score',
            'cv': sklearn.model_selection.KFold(n_splits=3, shuffle=True),  # gs__splitter
            'verbose': 1,
            'pre_dispatch': 1,
            'random_state': None,
            'error_score': np.nan,
            'return_train_score': True,
        },
        'my_gs_2': {
            'flag': False,
        }
    },
    'dataset': {
        'train': {
            'class': None,
            'steps': [
                # ('load_cache', {'func': None, 'prefix': None}),
                ('get', {'func': None, 'filename': 'data/train.csv'}),
                ('preprocess', {'func': None, 'categor_names': [], 'target_names': ['targets']}),
                ('info', {'func': None}),
                ('unify', {'func': None}),
                ('split', {'func': None,
                           'train_size': 0.7,  # 'split_train_size': 0.75,
                           'test_size': None,
                           'shuffle': False,
                           'stratify': None,
                           'random_state': None,}),
                ('dump_cache', {'func': None, 'prefix': None}),
            ],
        },
        'test': {
            'class': None,
            'steps': [
                ('get', {'func': None, 'filename': 'data/test.csv'}),
                ('preprocess', {'func': None, 'categor_names': [], 'target_names': ['targets']}),
                ('unify', {'func': None}),
            ],
        },
    },
}


if __name__ == '__main__':
    mlshell.run(params)

"""
create_pipeline => create
steps: None => steps: 'default'
 gs__runs => n_jobs
 gs__splitter => cv
 'cache_pipeline' => 'pipeline_cache'

params => conf
EVERYTHNG IN WIRKFLOW class
control-wise, each step as block.
# check if shuffle train>test with transformer
* Also two possibility, just one pipelines, metrics, gs configuration without name, or multiple
* Gs conf two possibilities: ``auto`` match with ppipeline_id or arbitrary name in fit
* Only one pipeline per conf file. otherwise too much information
* also add choice for gs kwargs, so can use alternative to RandomizedGS
* steps should be full with estimator, last step control via hp_grid
* generalization each public method, can choose data by data_id
    if split setted automative use 'test' or 'train' based on logic, or better "subtrain", "subtest" key specify
* del_duplicates remove, unify two possibility True and dic (wright logic for future)
* GetData, DataPreprocessor also in conf.py, they are pretty the same (classes => custom_classes)
* unifier
    possible to disable, then should be in DataPreprocesso (internally run anyway to get ind_names, hide unify_data in set_data)
* worflow control
    make possibility get static function
* allow multiple data
* all classes and main subfunctions (load, save, unify, validate, predict) should be replacable
    for predict need separate unifier
    user can just inheret with changes ans set classes here
* create default run.py, and conf.py as executable under if `__main__`
    if Get Started say about both possibilities.
* conf.py need separate docs. in Concept only concepts.
* unify add del_duplicates, cache kw_args
* rewrite GetData argument kwargs only
*  ** kw_args
* not sure about cache, separate o now, better separate i think
possibility to be full independent from sklearn

test:
* rewrite dic_flatter
* pipeline allow steps (if don`t need ind_names)
* possibility use built-in str name!!
"""


""" [deprecated]

params = {
    'pipeline': {
        'estimator': estimator,
        'type': 'regressor',
        'cache': None,
    },
    'metrics': {
        'score': (sklearn.metrics.r2_score, {'greater_is_better': True}),
        'mae': (sklearn.metrics.mean_absolute_error, {'greater_is_better': False}),
        'mse': (sklearn.metrics.mean_squared_error, {'greater_is_better': False, 'squared': False}),
    },
    'gs': {
        'flag': True,
        'hp_grid': hp_grid,
        'n_iter': None,  # 'gs__runs'
        'scoring': ['score', 'mae', 'mse'],
        'n_jobs': 1,
        'refit':'score',
        'cv' : sklearn.model_selection.KFold(n_splits=3, shuffle=True),  # gs__splitter
        'verbose' : 1,
        'pre_dispatch':1,
        'random_state':None,
        'error_score' : np.nan,
        'return_train_score': True,
    },
    'data': {
        'train': {
            'get':
                {'filiname': 'data/train.csv'},
            'unify':{
                'del_duplicates': False,
                'cache': None,
            },
            'split':
                {'train_size': 0.7,  # 'split_train_size': 0.75,
                 'test_size':None,
                 'shuffle': False,
                 'stratify': None,
                 'random_state':None,
                 },
        },
        'test':{
            'get':
                {'filiname': 'data/test.csv'},
            'unify': {
                'del_duplicates': False,
                'cache': None,
            },

        },
    },
}
"""