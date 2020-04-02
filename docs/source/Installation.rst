Installation Guide
==================

.. contents:: **Contents**
    :depth: 1
    :local:
    :backlinks: none

MLshell is compatible with: Python 3.6+.

PyPi
~~~~
|PyPI|

command to install package latest version:

.. code-block:: none

    pip install -U mlshell

command to install version with additional dev dependencies:

.. code-block:: none

    pip install -U mlshell[dev]


Docker hub
~~~~~~~~~~
|Docker|

command to create container with pre-installed package and run shell:

.. code-block:: none

    docker run -it nizaevka/mlshell


.. |PyPI| image:: https://img.shields.io/pypi/v/mlshell.svg
   :target: https://pypi.org/project/mlshell/

.. |Docker| image:: https://img.shields.io/docker/pulls/nizaevka/mlshell
   :target: https://hub.docker.com/r/nizaevka/mlshell/tags