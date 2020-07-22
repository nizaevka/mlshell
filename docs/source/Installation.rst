Installation Guide
==================

.. contents:: **Contents**
    :depth: 1
    :local:
    :backlinks: none


|project| is compatible with: Python 3.6+.

PyPi
~~~~
|PyPI|

command to install package latest version:

.. parsed-literal::

    pip install -U |project|

command to install version with additional dev dependencies:

.. parsed-literal::

    pip install -U |project|\[dev]


Docker hub
~~~~~~~~~~
|Docker|

command to create container with pre-installed package and run shell:

.. parsed-literal::

    docker run -it |author|/|project|

.. not easy to substitute author/project here.
    https://stackoverflow.com/questions/20513972/restructured-text-sphinx-substitution-in-a-file-name
.. |PyPI| image:: https://img.shields.io/pypi/v/mlshell.svg
   :target: https://pypi.org/project/mlshell/

.. |Docker| image:: https://img.shields.io/docker/pulls/nizaevka/mlshell
   :target: https://hub.docker.com/r/nizaevka/mlshell/tags
