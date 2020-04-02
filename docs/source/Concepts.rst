Concepts
========


.. image:: ./_static/images/workflow.png
    :width: 1000
    :alt: error

.. contents:: **Contents**
    :depth: 1
    :local:
    :backlinks: none

Pipeline structure
^^^^^^^^^^^^^^^^^^

- Unified pipeline based on sklearn.Pipeline.
- Data scientist can use it for broad range of tasks with minimal changes in code.
- It provides stable cross-validation scheme and prevents data leaks.
- Every parameter at each step can be tuned in GS CV.

Embedded universal pipeline for grid search:

.. code-block:: none

    default_steps = [
        ('pass_custom',      pass custom params to custom scorer in GS)
        ('select_rows',      delete rows (outliers/anomalies))
        ('process_parallel',
            ('pipeline_categoric',
               ('select_columns',      select categorical(and binary) subcolumns)
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
Set corresponding parameters in conf.py hp_grid dictionary to overwrite default.

If necessary engineer can redefine ``CreateDefaultPipeline`` class in conf.py by specify ``pipeline`` parameter.

Data requirements
^^^^^^^^^^^^^^^^^

Engineer have to write:

* data wrapper from database.
* data preprocessor to prepare dataframe for Workflow class.

.. note::

    dataframe should have columns={'targets', 'feature_<name>', 'feature_categor_<name>'}

        * 'feature_categor_<name>': any dtype.

        order is not important (include binary).

        * 'feature_<name>': any numeric dtype (should support float(val): np.issubdtype(type(val), np.number))

        order is important.

        * 'targets': any dtype.

        for classification ``targets`` should be binary, ordinalencoded.
        | positive label should be > others in np.unique(``targets``) sort.

Data unification
^^^^^^^^^^^^^^^^

Before use with pipeline input data should be unified with ``Workflow.unify_data()`` method.

Internally:

- categorical features are OrdinalEncoded.
- categorical gaps are imputed with .fillna(value='unknown').
- numeric gaps are imputed with .fillna(value=np.nan).
- dataframe is casted to .astype(np.float64).

Configuration file example
^^^^^^^^^^^^^^^^^^^^^^^^^^

Data scientist can set all workflow parameters through one configuration file.

``conf.py`` should specify at least:

- main estimator.
- cross-validation splitter and split ratio.
- metrics to evaluate (metric with name 'score' will use to sort results in GS).
- grid search parameters ``hp_grid``.

.. code-block:: python

    params = {
        'estimator_type': 'classifier',
        'main_estimator': main_estimator,
        'metrics': {
            'score': (sklearn.metrics.roc_auc_score, {'greater_is_better': True, 'needs_proba': True}),
            'precision': (sklearn.metrics.precision_score, {'greater_is_better': True, 'zero_division': 0}),
            'custom': (custom_score_metric, {'greater_is_better': True}),
            'confusion matrix': (sklearn.metrics.confusion_matrix, {'labels': [1, 0]}, False),
            'classification report': (sklearn.metrics.classification_report, {'output_dict': True, 'zero_division': 0}, False),
        },
        'split_train_size': 0.7,
        'cv_splitter': sklearn.model_selection.KFold(n_splits=3, shuffle=True),
        'hp_grid': hp_grid,

see `default params <Default-configuration.html#mlshell.default.DEFAULT_PARAMS>`_ for full list.

Split
^^^^^

- ``train`` data will be split on ``subtrain`` and ``validation`` datasets by sklearn.model_selection.train_test_split with ``split_train_size`` proportion.
- ``subtrain`` will be passed to gs cross-validation with ``cv_splitter``.
- ``validation`` will be used to evaluate metrics on best model from gs.
- If ``split_train_size`` = 1, gs use whole dataset and ``validation``=``train`` for compatibility.


Metrics
^^^^^^^

Specify metrics in ``metrics`` dictionary.
One with ``score`` key  will be used for best model selection in grid search.
Metric dict value should contain tuple/list with at most three entity:

    * 0 index: metric callback.
    * 1 index: make_scorer or metric function kwargs.
    * | 2 index: boolean flag, pass metric in grid search to evaluation.
      | if True: create make_scorer from metric and pass in gs.
      | else: evaluate metric only on holdout validation set (confusion matrix for example).


Hyperparameters grid
^^^^^^^^^^^^^^^^^^^^

- Each parameter(hp) at every pipeline step can be tuned in GS.
- Set one value for parameter to overwrite default.
- Set multiple values to proceed GS on that parameter.

.. code-block:: python

    hp_grid = {
        # custom scorer param
        'pass_custom__kw_args': [{'param_a': 0, 'param_b': 0}, ],
        # transformers param
        'select_rows__kw_args': [{}],
        'process_parallel__pipeline_numeric__impute__gaps__strategy': ['constant'],
        'process_parallel__pipeline_numeric__transform_normal__skip': [True],
        'process_parallel__pipeline_numeric__scale_column_wise__quantile_range': [(1, 100)],
        'process_parallel__pipeline_numeric__add_polynomial__degree': [3],
        'process_parallel__pipeline_numeric__compose_columns__discretize__n_bins': [5],
        'select_columns__estimator__skip': [True],
        'reduce_dimension__skip': [True],
        # regressor only
        'estimate__transformer': [None, sklearn.preprocessing.FunctionTransformer(func=np.log, inverse_func=np.exp)],
        # estimator params
        'estimate__regressor__n_estimators': np.linspace(50, 500, 10, dtype=int),
        'estimate__regressor__num_leaves' :[2 ** i for i in range(1, 6 + 1)],
        'estimate__classifier__min_child_samples': np.linspace(1, 100, 10, dtype=int),
        'estimate__regressor__max_depth': np.linspace(1, 10, 10, dtype=int),
        # classifier only
        'estimate__apply_threshold__threshold': [0.1],
    }

.. note::

    'estimate__transformer' applied only to regressors for target transformation, ignored if classifier.

    'estimate__apply_threshold__threshold' applied only to classifier for threshold tuning (default = 0.5).

Grid Search
^^^^^^^^^^^
* if ``gs_flag`` is True:

    | Will run gridsearch(GS) and fit estimator with the best parameters.
    | sklearn.model_selection.RandomizedSearchCV is used by default.

* else:

    | If any param specified in hp_grid with sequence:
    | pipeline will be fitted with the value on the zero position of parameter range, otherwise  default value.

.. note::

    | Internal hps optimization:
    | optimizer = sklearn.model_selection.RandomizedSearchCV(
    |   pipeline, ``hp_grid``, scoring=scorers, n_iter= ``n_iter``,
    |   n_jobs= ``n_jobs``, pre_dispatch=pre_dispatch,
    |   refit='score', cv= ``cv_splitter``, verbose= ``gs_verbose``, error_score=np.nan,
    |   return_train_score=True).fit(x_subtrain, y_subtrain, ** ``estimator_fit_params``)

    Scorers generally are made from ``metrics`` (see `Classification threshold` below for special cases).

    | If ``runs`` is None, there will be as much runs as hps combinations defined in hp_grid.
    | If any of the params specified in hp_grid with distribution: ``runs`` should be specified.

    | The ``n_jobs`` control number of parallel CV, and dataset is copied in RAM pre_dispatch times.
    | pre_dispatch = max(2, ``n_jobs``) if ``n_jobs`` else 1 (spawn jobs in advance for faster queueing)
    | If ``n_jobs`` = -1, dataset is copied in RAM hp-combinations times (ignores pre_dispatch limit).

.. warning::

    Disable np.random.seed() if use sampling from distribution for any parameter.

Classification threshold
^^^^^^^^^^^^^^^^^^^^^^^^^

For classification task it is possible to tune classification threshold ``th_`` on CV.
If positive class probability P(positive label) = 1 - P(negative label) > ``th_`` for some sample,
classifier set pos_label for this sample, otherwise negative_label.

In general,

    * we can consider ``th_`` as hp,
    * each fold in CV has it own best ``th_``, we try to find value good for all folds,
    * ``th_`` search range can be got from ROC curve on classifier`s predict_proba.

Mlshell support multiple strategy for ``th_`` tuning:

.. image:: ./_static/images/th_strategy.png
    :width: 1000
    :alt: error

.. note::

    (0) Don't use ``th_`` (common case).

        * Not all classificator provide predict_proba (SVM).
        * We can use f1, logloss metrics.
        * If necessary you can dynamically pass params in custom scorer function to tune them in CV (through 'pass_custom__kw_args' step in hp_grid).

    (1) First GS best hps with CV, then GS best ``th_`` (common).

        * For GS hps by default used auc-roc as score.
        * For GS ``th_`` used main score.


    (2) Use additional step in pipeline (meta-estimator) to GS ``th_`` in predefined range (experimental).

        * Tune ``th_`` on a par with other hps.
        * ``th_`` range should be unknown in advance:

            (2.1) set in arbitrary in hp_grid.

            (2.2) take typical values from ROC curve OOF.

    (3) While GS best hps with CV, select best ``th_`` for each fold separately (experimental).

        * For current hps combination maximize tpr/(tpr+fpr) on each fold by ``th_``.
        * | Although there will different best ``th_`` on folds,
          | the generalizing ability of classifier might be better.
        * Then select single overall best ``th_`` on GS with main score.

    In 1/2.2/3 strategies ``th_`` range came from ROC curve on OOF prediction_proba.

    By default tpr/(tpr+fpr) is maximized, then points are linear sampled from [max/100,max*2] with [0,1] limits.

    Engineer can specify number of samples 'th_points_number' (default=100) and plot roc_curve 'th_plot_flag'.

``th_`` range extract example:

.. image:: ./_static/images/th_.png
  :width: 1000
  :alt: error

.. warning::
    | Currently, only binary classification is supported.
    | Be carefull with experimental features.
    | TimeSeriesSplit OOF don`t provide the first fold in th_strategy (1)-(3).


Workflow
^^^^^^^^

- Mlshell is production ready.
- Data scientist can control the workflow through a script or a notebook.

see `Get started <Get-started.html>`_ for full workflow file example.

Project structure
^^^^^^^^^^^^^^^^^

.. code-block:: none

    |_project/
        ** input **
        |__ conf.py
        |__ run.py
        |__ EDA.ipynb
        |__ data/
            ~~ could be remote db ~~
            |__ train.csv
            |__ test.csv
        ** output autogenerated **
        |__ results/
            |__ models/
                ~~ dump fitted models and predictions ~~
                |__ <params_hash>_<train_data_hash>_dump.model
                |__ <params_hash>_<new_data_hash>_predictions.csv
            |__ runs/
                ~~ dump all GS runs result ~~
                |__ <timestamp>_runs.csv
            |__ run_logs/
                ~~ script logs ~~
                |__ <logger_name>_<logger_level>.log
            |__ ipython_logs/
                ~~ notebook logs ~~
                |__ <logger_name>_<logger_level>.log
            |__ temp/
                ~~ cache for sklearn.pipeline.Pipeline(memory=<./temp>) ~~

Results
^^^^^^^

**runs.csv**
~~~~~~~~~~~~

Runs of each GS workflow will be dumped in <timestamp>_runs.csv.

see `dump_runs method <_pythonapi/mlshell.Workflow.html#mlshell.Workflow.dump_runs>`_ for details.

``*_runs.csv`` files could be merge in dataframe for further analyse:

.. code-block:: python

    from os import listdir
    files = [f for f in listdir('results/runs/') if 'runs.csv' in f]
    df_lis = list(range(len(files)))
    for i,f in enumerate(files):
        if '.csv' not in f:
            continue
        try:
            df_lis[i]=pd.read_csv("results/runs/" + f, sep=",", header=0)
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
~~~~~~~~

- If it possible, logger files will be called the same as workflow start file.
- There are 7 levels of logging files:

    * critical
        reset on logger creation.
    * error
        reset on logger creation.
    * warning
        reset on logger creation.
    * minimal
        cumulative.
        only score for best run in gs.
    * info
        cumulative.
        workflow information.
    * debug
        reset on logger creation.
        detailed workflow information.
    * test
        only for test purposes.

see `logger configuration <_modules/mlshell/logger.html>`_ for details.

**gui**
~~~~~~~

For small dataset it is reasonable to visualize unravel score per samples.

Mlshell provides experimental gui:

* for regression:

    * dynamical plot main (r2) score (figure right axis),
    * dynamical plots of normalized mae/mse score sum on adding samples (left axis),
    * residuals scatter on user-defined base_plot (target column for example).

* for classification:

    * dynamical plot main (precision) score (figure right axis),
    * TP/FP/FN scatters on user-defined base_plot (diagonal line for example).

Also sliders for grid search range available, model retrained and make predict at each slider change (except threshold)

Engineer should specify base_plot in classes.DataPreprocessor.

GUI.plot(base_sort=True) method have flag ``base_sort`` to turn on/off sorting of base_plot vector (default=False).

see `Examples <Examples.html>`_.