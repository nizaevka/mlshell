Python Quick Start
==================

.. contents:: **Contents**
    :depth: 1
    :local:
    :backlinks: none

Data preparation
~~~~~~~~~~~~~~~~
- Create workdir.
- Create conf.py in workdir with workflow configuration.
- Implement custom GetData to read the raw data from db.
- Implement custom DataPreprocessor to transform data in consistence with Workflow class.
- Create a run script or a notebook to control workflow.

Code
~~~~

:github:`github repo </examples/simplest>`

Simplest conf.py example:

.. literalinclude:: /../../examples/simplest/conf.py
   :language: python
   :lineno-match:

Control workflow example:

.. literalinclude:: /../../examples/regression/run.py
   :language: python
   :end-before: # read and preprocess test data
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

Next
~~~~

For detailed examples please follow:

- `regression <Examples.html#regression>`_
- `classification <Examples.html#classification>`_


| For Workflow data requirements see `Workflow class <_pythonapi/mlshell.Workflow.html#mlshell.Workflow>`_ Note.
| For full list of conf.py params see `DEFAULT_PARAMS <Default-configuration.html#mlshell.default.DEFAULT_PARAMS>`_  description.
