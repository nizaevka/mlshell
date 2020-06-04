Configuration
=============

Workflow configuration is set with python dictionary.
It could be passed to the `mlshell.run()` function or some user-defined handler, where
workflow class is built and its endpoints are called.
TODO: see "" for details.


# картинка

.. contents:: **Contents**
    :depth: 1
    :local:
    :backlinks: none



Default workflow class have set of built-in functions:

* fit
* optimize
* validate
* predict
* dump
* plot

Specify kwargs

.. note::
    It is also possible to reassign any built-in functions via `func` argument,
    or add new arbitrary function to utilize in `workflow``steps` after.

.. note::
    For each endpoint configuration there is additional `global` field.

so we can resolve None by key name
        # if fit_2 can`t resolve => set from built in or new

.. note::
    * 'func': None
    *

**Grid search**
^^^^^^^^^^^^^^^

