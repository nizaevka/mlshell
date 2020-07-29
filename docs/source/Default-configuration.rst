Default configuration
---------------------

CNFG
****

.. literalinclude:: /../../src/mlshell/conf.py
    :language: python
    :start-at: import pycnfg
    :end-before: if __name__ == '__main__':

logger
******

.. literalinclude:: /../../src/mlshell/producers/logger.py
    :language: python
    :start-at: class LevelFilter(object):
    :end-before: class LoggerProducer(pycnfg.Producer):

.. Not bad
.. .. autodata:: mlshell.conf.CNFG
    :annotation:
.. .. pprint::
    mlshell.conf.CNFG

.. Problem, showed in name.
.. .. automodule:: mlshell.CNFG
    :members:
    :undoc-members:
    :show-inheritance:

.. Problem, showed in name.
.. .. autosummary::
    :toctree: _pythonapi/
    mlshell.CNFG

.. universal
.. .. exec::
    import pprint
    import mlshell
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(mlshell.CNFG)

.. https://stackoverflow.com/questions/27875455/displaying-dictionary-data-in-sphinx-documentation
