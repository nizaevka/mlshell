Examples
========

.. contents:: **Contents**
    :depth: 1
    :local:
    :backlinks: none

Regression
~~~~~~~~~~
`"Allstate claims severity" dataset <https://www.kaggle.com/c/allstate-claims-severity>`_

:github:`github repo </examples/regression>`


**conf.py**
^^^^^^^^^^^

.. literalinclude:: /../../examples/regression/conf.py
   :language: python

.. https://docutils.sourceforge.io/docs/ref/rst/directives.html#including-an-external-document-fragment
.. .. code-block:: python

**Results**
^^^^^^^^^^^

`info logs <_static/text/regression_info.log>`_

`minimal logs <_static/text/regression_minimal.log>`_

.. .. include:: ./_static/text/regressor_run_info.log
   :literal:
.. .. :download:`minimal log <./_static/text/regressor_run_minimal.log>`
.. **gui**
.. base_plot is sorted target

.. .. image:: ./_static/images/gui_regression.png
  :width: 1000
  :alt: error

Classification
~~~~~~~~~~~~~~
`"IEEE fraud detection" dataset <https://www.kaggle.com/c/ieee-fraud-detection>`_

:github:`github repo </examples/classification>`

**conf.py**
^^^^^^^^^^^

.. literalinclude:: /../../examples/classification/conf.py
   :language: python

**Results**
^^^^^^^^^^^

`info logs <_static/text/classification_info.log>`_

`minimal logs <_static/text/classification_minimal.log>`_

.. note::

    - In second optimization stage ``th_`` range came from ROC curve on OOF predictions for positive label.
     Number of points set in resolve_params configuration.
     Positive label set either in dataset configuration, or auto-resolved as last one in np.unique(targets).
    .. image:: ./_static/images/th_.png
        :width: 1000
        :alt: error
    As expected: ROC AUC not depends on threshold (as evaluates on predict_proba).
    Precision (and custom metric) depends on threshold. If positive class ``th_``
    close to 1, all samples classified as negative class (TP=0 and FP=0), so
    precision become ill-defined, score set to 0 as specified in 'zero_division'
    argument fro precision metric.

    - In third optimization stage, ``kw_args`` tunned for cutom score function.
    There could be arbitrary logic.


see `Concepts: classifier threshold <Concepts.html#classifier-threshold>`_ for details.

.. **gui**

.. base_plot is diagonal line

.. .. image:: ./_static/images/gui_classification.png
  :width: 1000
  :alt: error

.. see `Concepts: Results <Concepts.html#gui>`_ for details.

