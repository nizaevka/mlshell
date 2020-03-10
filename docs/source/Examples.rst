Examples
========

.. contents:: **Contents**
    :depth: 1
    :local:
    :backlinks: none

Regression
~~~~~~~~~~
`"Allstate claims severity" dataset <https://www.kaggle.com/c/allstate-claims-severity>`_

`example code regression <https://github.com/nizaevka/mlshell/tree/master/examples/regression>`_

**For grid search try the ideas:**

* estimator: linear vs lightgbm vs xgboost.
* features 'yeo-johnson' transformation.
* features quantile scaler.
* polynomial feature generation: degree 2 vs 3.
* target transformer: None vs np.log vs y**0.25.
* loss: mse vs mae vs `ln(cosh(x)) <https://www.kaggle.com/c/allstate-claims-severity/discussion/24520>`_.
* estimator hyperparameters.

If time is limited:

    * use subset of data (set `rows_limit`, set `random_skip` to randomize subset)
    * use greedy approach:

        * find approximately reasonable estimator` hps
        * try ideas
        * for the best refine estimator hps

**conf.py example:**

* use target_transformer instead of default.
* GS add_polynomial__degree and transform_normal__skip.
* 10000 rows subset of data.

.. code-block:: python

    import numpy as np
    import sklearn
    import lightgbm
    import xgboost


    # choose estimator
    main_estimator = [
        # sklearn.linear_model.SGDRegressor(penalty= 'elasticnet',
        #                                   l1_ratio=0.01,
        #                                   alpha=0.01,
        #                                   shuffle=False,
        #                                   max_iter=1000,
        #                                   early_stopping=True,
        #                                   learning_rate='invscaling',
        #                                   power_t=0.25,
        #                                   eta0=0.01,
        #                                   verbose=1),
        # ensemble.RandomForestRegressor(n_estimators=300, max_depth=10, max_features=0.1, criterion='mae'),
        lightgbm.LGBMRegressor(objective='fair',
                               num_leaves=2,
                               min_data_in_leaf=1,
                               n_estimators=250,
                               max_depth=-1,
                               silent=False)
        # xgboost.XGBRegressor(objective="reg:squarederror", **{
        #     'min_child_weight': 1,
        #     'eta': 0.01,
        #     'n_estimators': 500,
        #     'colsample_bytree': 0.5,
        #     'max_depth': 12,
        #     'subsample': 0.8,
        #     'alpha': 1,
        #     'gamma': 1,
        #     'silent': 1,
        #     'verbose_eval': True,
        #     'seed': 42,
        #     })
        ][0]


    # define hyperparameters (hp) to cv
    def target_func(y):
        return y**0.25


    def target_inverse_func(y):
        return y**4


    # create target transformers (don`t use lambda function)
    target_transformer = sklearn.preprocessing.FunctionTransformer(func=target_func, inverse_func=target_inverse_func, validate=False, check_inverse=True)
    target_transformer_2 = sklearn.preprocessing.FunctionTransformer(func=np.log, inverse_func=np.exp, validate=False, check_inverse=True)


    # set ranges for hp
    hp_grid = {
        # 'process_parallel__pipeline_numeric__impute__gaps__strategy': ['median', 'constant'],
        'process_parallel__pipeline_numeric__transform_normal__skip': [True, False],
        # 'process_parallel__pipeline_numeric__scale_column_wise__quantile_range': [(0, 100), (1, 99)],
        'process_parallel__pipeline_numeric__add_polynomial__degree': [2, 3],
        'estimate__transformer': [target_transformer],

        # # lgbm
        # 'estimate__regressor__n_estimators': np.linspace(50, 1000, 10, dtype=int),
        # 'estimate__regressor__num_leaves': [2**i for i in range(1, 6 + 1)],
        # 'estimate__regressor__min_data_in_leaf': np.linspace(10, 100, 10, dtype=int),
        # 'estimate__regressor__max_depth': np.linspace(1, 30, 10, dtype=int),
    }


    # set workflow params
    params = {
        'estimator_type': 'regressor',
        'main_estimator': main_estimator,
        'cv_splitter': sklearn.model_selection.KFold(n_splits=3, shuffle=True),
        'metrics': {
            'score': (sklearn.metrics.mean_absolute_error, False),
            'r2': (sklearn.metrics.r2_score, True),
            'mse': (sklearn.metrics.mean_squared_error, False),
        },
        'split_train_size': 0.7,
        'hp_grid': hp_grid,
        'gs_flag': True,
        'del_duplicates': False,
        'debug_pipeline': False,
        'isneed_cache': False,
        'cache_update': False,
        'gs_verbose': 1000,
        'n_jobs': 1,
        'isneeddump': False,
        'runs': None,
        'plot_analysis': False,

        'train_file': 'data/train.csv',
        'test_file': 'data/test.csv',
        'rows_limit': 10000,
        'random_skip': False,
    }



**Results:**

.. image:: ./_static/images/result_regression.jpg
  :width: 1000
  :alt: error

Classification
~~~~~~~~~~~~~~
`"IEEE fraud detection" dataset <https://www.kaggle.com/c/ieee-fraud-detection>`_

`example code classifiction <https://github.com/nizaevka/mlshell/tree/master/examples/classification>`_
