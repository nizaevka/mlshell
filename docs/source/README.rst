Documentation
=============

Documentation for |project| is generated using `Sphinx <http://www.sphinx-doc.org/>`__.
``Python API`` is generated automatically from docstrings. After each commit to
``master`` branch, documentation is updated and published to :readthedocs:`Read the Docs<>`.

Build
-----

You can build the documentation locally, in ``docs`` dir run:

.. code:: sh

    pip install -r requirements.txt
    make html
