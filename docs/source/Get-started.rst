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
- Implement custom GetData to read the raw data from db (for example see  :github:`classes.py </examples/regression/classes.py>`).
- Implement custom DataPreprocessor to transform readed data in consistence with Workflow class.

    * for example see :github:`classes.py </examples/regression/classes.py>`.
    * for Workflow data requirements see Note for `Workflow class <_pythonapi/mlshell.Workflow.html#mlshell.Workflow>`_.
- Create a run script or a notebook to control workflow.

Code example
~~~~~~~~~~~~

.. literalinclude:: /../../examples/regression/run.py
   :language: python
   :end-before: # read and preprocess new data
   :lineno-match:

to make prediction on new data:

.. literalinclude:: /../../examples/regression/run.py
   :language: python
   :start-after: wf.load(file)
   :end-before: # generate param for gui
   :lineno-match:

to plot "score vs samples" gui on train data:

.. literalinclude:: /../../examples/regression/run.py
   :language: python
   :start-after: wf.predict
   :lineno-match:

For detailed example please follow:

- `regression <Examples.html#regression>`_
- `classification <Examples.html#classification>`_
