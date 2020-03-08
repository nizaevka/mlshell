Python Quick Start
==================

.. contents:: **Contents**
    :depth: 1
    :local:
    :backlinks: none

Data preparation
~~~~~~~~~~~~~~~~
- Create workdir.
- Create conf.py in wokdir with workflow configuration (for example see `Examples <Examples.html>`_).
- Implement custom GetData to read the raw data from db (for example see `classes.py <https://github.com/nizaevka/mlshell/examples/regression/classes.py>`_).
- Implement custom DataPreprocessor to transform readed data in consistence with Workflow class.

    * for example see `classes.py <https://github.com/nizaevka/mlshell/examples/regression/classes.py>`_.
    * for Workflow data requirements see Note for `Workflow class <_pythonapi/mlshell.Workflow.html#mlshell.Workflow>`_.
- Create a run script or a notebook to control workflow.

Code example
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import mlshell
    from classes import GetData, DataPreprocessor

    # find path
    project_path, script_name = mlshell.find_path()
    # create logger
    logger = mlshell.logger.CreateLogger(f"{project_path}/logs_{script_name}").logger

    # get params from conf.py in workdir
    gp = mlshell.GetParams(project_path)

    # get data from db (project specific)
    gd = GetData(logger)
    gd.get_data(filename=gp.params['train_file'], rows_limit=gp.params['rows_limit'], random_skip=gp.params['random_skip'])

    # prepare data for analyse (project specific)
    pp = DataPreprocessor(logger, gd.raw)

    # initialize object of Workflow class (encode/unify data included)
    wf = mlshell.Workflow(project_path, logger, pp.data, params=gp.params)

    # EDA on whole data
    # wf.before_split_analyze()

    # create pipeline
    wf.create_pipeline()  # self.estimator

    # split data
    wf.split()  # => self.train, self.test

    # fit pipeline on train (tune hp if GS_flag=True)
    wf.fit(gs_flag=gp.params['gs_flag'])

    # test prediction
    wf.validate()

    # dump on disk
    file = wf.dump()

to make prediction on new data:

.. code-block:: python

    # load from disk
    wf.load(file)

    # read and preprocess new data
    gd2 = classes.GetData(logger)
    gd2.get_data(filename=gp.params['test_file'], rows_limit=gp.params['rows_limit'])
    pp2 = classes.DataPreprocessor(logger, gd2.raw)

    # make prediction for new data
    wf.predict(pp2.data, pp2.raw_targets_names, pp2.raw_index_names)

to plot "score vs samples" with hp sliders for train data:

.. code-block:: python

    # prepare param for gui on workflow
    wf.gen_gui_params()

    # init gui object
    gui = GUI(logger, pp.base_plot, wf.gui_params)

    # plot results
    gui.plot(isplot=True)

For detailed example please follow:

- `regression <Examples.html#regression>`_
- `classification <Examples.html#classification>`_
