Python Quick Start
==================

.. contents:: **Contents**
    :depth: 1
    :local:
    :backlinks: none

Project
~~~~~~~
- Create workdir.
- Add conf.py with configuration dictionary CNFG.
- Pass CNFG to :func:`pycnfg.run` .

Code
~~~~

:github:`github repo </examples/simplest>`

conf.py:

.. literalinclude:: /../../examples/simplest/conf.py
   :language: python

Shortened version using :mod:`pycnfg` features :

.. literalinclude:: /../../examples/simplest/conf_short.py
   :language: python
   :start-at: CNFG

Next
~~~~

For more complex examples please follow:

- `regression <Examples.html#regression>`_
- `classification <Examples.html#classification>`_

| :doc:`User Guide <Concepts>`
| :ref:`default CNFG <Default-configuration:CNFG>`
