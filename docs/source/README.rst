Documentation
=============

Documentation for MLshell is generated using `Sphinx <http://www.sphinx-doc.org/>`__.

List of Python API and their descriptions in `Python-API.rst <./Python-API.rst>`__
is generated automatically from docstrings.

After each commit on ``master``, documentation is updated and published to `Read the Docs <https://mlshell.readthedocs.io/>`__.

Build
-----

You can build the documentation locally.

.. code:: sh

    pip install -r requirements.txt
    make html
