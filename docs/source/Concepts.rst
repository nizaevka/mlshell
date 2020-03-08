Features
========

.. image:: _images/workflow.jpg
  :width: 1000
  :alt: Alternative text

.. contents:: **Contents**
    :depth: 1
    :local:
    :backlinks: none

Pipeline structure
^^^^^^^^^^^^^^^^^^

- Unified pipeline based on sklearn.Pipeline.
- Data scientist can use it for broad range of tasks with minimal changes in code.
- It provides stable cross-validation scheme and prevents data leaks.
- Each parameter at every step can be tuned in GS CV.

Embedded universal pipeline for grid search:

.. code-block:: none

    default_steps = [
        ('pass_custom',      pass custom params to custom scorer in GS)
        ('select_rows',      delete rows (outlier/anomalies))
        ('process_parallel',
            ('pipeline_categoric',
               ('select_columns',      select categorical(and binary) subcolumns )
               ('encode_onehot',       OneHot encoder)
            )
            ('pipeline_numeric',
                ('select_columns',     select numeric subcolumns)
                ('impute',
                    ('indicators',     impute indicators)
                    ('gaps',           impute gaps)
                )
                ('transform_normal',   yeo-johnson features transforamtion)
                ('scale_row_wise',     row_wise transformation)
                ('scale_column_wise',  column_wise transformation)
                ('add_polynomial',     add polynomial features)
                ('compose_columns',
                    ("discretize",     discretize columns)
                )
            )
        )
        ('select_columns',   model-wise feature selection)
        ('reduce_dimension', Factor analyze feature selection/transformation)
        ('estimate',         target transform)
    ]


see `CreateDefaultPipeline source <_modules/mlshell/default.html#CreateDefaultPipeline>`_ for details.

By default only OneHot encoder and imputer (gaps and indicators) are activated.
Set corresopnding parameters in conf.py hp_grid dictionary to overwrite default.

Configuration file example
^^^^^^^^^^^^^^^^^^^^^^^^^^

Data scientist can set all workflow parameters through one configuration file.

`conf.py` should specify at least:

- main estimator
- cross-validation splitter and split ratio
- metrics to evaluate (metric with name 'score' will use to sort results in GS)
- grid search parameters `hp_grid`

.. code-block:: python

    params = {
        'estimator_type': "regressor",
        'main_estimator': lightgbm.LGBMRegressor(),
        'cv_splitter': sklearn.model_selection.KFold(n_splits=3, shuffle=True),
        'metrics': {
            'score': (sklearn.metrics.r2_score, True),
            'mae': (sklearn.metrics.mean_absolute_error, False)
        },
        'split_train_size': 0.7,
        'hp_grid': hp_grid,
        'gs_flag':True,
    }

see `default params <Default-configuration.html#mlshell.default.DEFAULT_PARAMS>`_ for full list.

Hyperparameters grid
^^^^^^^^^^^^^^^^^^^^

- Every parameter of at every pipeline step can be tuned in GS.
- Set one value for param to use instead of default.
- Set multiple values to proceed GS on that param.

.. code-block:: python

    hp_grid = {
        # custom any params to use in custom scorer
        'pass_custom__kw_args': [{'param_a': 0, 'param_b': 0}, ],
        'select_rows__kw_args': [{}],
        'process_parallel__pipeline_numeric__impute__gaps__strategy': ['constant'],
        'process_parallel__pipeline_numeric__transform_normal__skip': [True],
        'process_parallel__pipeline_numeric__scale_column_wise__quantile_range': [(1, 100)],
        'process_parallel__pipeline_numeric__add_polynomial__degree': [3],
        'process_parallel__pipeline_numeric__compose_columns__discretize__n_bins': [5],
        'select_columns__estimator__skip': [True],
        'reduce_dimension__skip': [True],
        'estimate__transformer': [None, sklearn.preprocessing.FunctionTransformer(func=np.log, inverse_func=np.exp)],
        # estimator params
        'estimate__regressor__n_estimators': np.linspace(50, 500, 10, dtype=int),
        'estimate__regressor__num_leaves' :[2 ** i for i in range(1, 6 + 1)],
        'estimate__regressor__min_data_in_leaf': np.linspace(10, 100, 10, dtype=int),
        'estimate__regressor__max_depth': np.linspace(1, 10, 10, dtype=int),
    }

Grid Search
^^^^^^^^^^^

* if gs_flag is True:

    Run gridsearch and fit estimator with the best parameters.
* else:

    If any of the params specified in hp_grid:
    pipeline will be fitted with the value on the zero position of list, default otherwise.

Workflow
^^^^^^^^

- Mlshell is production ready.
- Data scientist can control the workflow through script or notebook.

see `Get started <Get-started.html>`_ for full worflow file example.

Project structure
^^^^^^^^^^^^^^^^^

.. code-block:: none

    project/
    # input
    - conf.py
    - run.py
    - EDA.ipynb
    + data/
        - train.csv
        - test.csv
    # output
    + models/
        # autocreated to dump fitted models and predictions
        <params_hash>_<train_data_hash>_dump.model
        <params_hash>_<new_data_hash>_predictions.csv
    + runs/
        # autocreated to dump all GS runs result
        <timestamp>_runs.csv
    + run_logs/
        # autocreated to save logs
        <logger_name>_<logger_level>.log
    + ipython_logs/
        # autocreated to save notebook logs
        <logger_name>_<logger_level>.log

Results
^^^^^^^

**runs.csv**

Every GS run <timestamp>_runs.csv will be dumped.

see `dump_runs method <_pythonapi/mlshell.Workflow.html#mlshell.Workflow.dump_runs>`_ for details.

*_runs.csv files could be merge in dataframe for further analyse.

.. code-block:: python

    from os import listdir
    files = [f for f in listdir('runs/') if 'runs.csv' in f]
    df_lis = list(range(len(files)))
    for i,f in enumerate(files):
        if '.csv' not in f:
            continue
        try:
            df_lis[i]=pd.read_csv("runs/" + f, sep=",", header=0)
            print(f, df_lis[i].shape, df_lis[i]['data_hash'][0], df_lis[i]['params_hash'][0])
        except Exception as e:
            print(e)
            continue

    df=pd.concat(df_lis,axis=0,sort=False).reset_index()
    # groupby data hash
    df.groupby('data_hash').size()
    # groupby estumator type
    df.groupby('estimator_name').size()

**logs**

- If it possible logger would be called the same as workflow file.
- There are 5 levels of logging:

    * critical (results of validation)
    * error
    * warning
    * info
    * debug
see `logger configuration source <_modules/mlshell/logger.html>`_ for details.
