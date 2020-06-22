"""
The module includes auxiliary utilities.

"""

import os
import sys

from IPython import get_ipython
import mlshell.pycnfg as pycnfg

__all__ = ['find_path', 'run']


def find_path(filepath=None):
    """Get fullpath and name of main script.

    Args:
        filepath (str): path to main script (default=None)
            if None and Ipython get from workdir,
            | if None and standart get from sys.argv,
            | if sys.argv empty, get from working directory.

    Returns:
        (tuple): tuple containing:

        - project_dir (str): full path to start script directory
        - script_name (str): name of main script if not Ipython
            else 'ipython'

    """
    def is_ipython():
        """Return True if ipython else False"""
        try:
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':
                # Jupyter notebook or qtconsole.
                return True
            elif shell == 'TerminalInteractiveShell':
                # Terminal running IPython.
                return True
            else:
                # Other type (?).
                return False
        except NameError:
            # Probably standard Python interpreter
            return False
    if filepath is not None:
        temp = filepath.replace('\\', '/').split('/')
        project_dir = '/'.join(temp[:-1])
        script_name = temp[-1][:-3]
    elif is_ipython():
        project_dir = os.getcwd().replace('\\', '/')
        script_name = 'ipython'
    else:
        # sys.argv provide script_name but not work in Ipython
        # for example ['path/run.py', '55']
        ext_size = 3  # '.py'
        script_name = sys.argv[0].replace('\\', '/').split('/')[-1][:-ext_size]  # run
        project_dir = sys.argv[0].replace('\\', '/')[:-len(script_name)-ext_size]

    return project_dir, script_name


def run(conf, default_conf=None, project_path=None, logger=None):
    """Wrapper over configuration handler.

    Parameters
    ----------
    conf : dict
        Configuration to pass in `pycnfg.Handler().read()`.
    default_conf : None, dict, optional (default=None)
        Default configurations to pass in `ConfHandler().read()`.
    project_path: str, optional (default='')
        Absolute path to current project dir.
        If None, auto detected by `pycnfg.find_path()`.
    logger : None, logger object (default=None)
        If None, `pycnfg.GetLogger()` will be used with script name.

    Returns
    -------
    objects : dict {'configuration_id': object}.
        Dict of objects created by execution all configurations.

    See Also
    --------
    :class:`pycnfg.Handler`:
        Reads configurations, executes steps.
    :class:`pycnfg.GetLogger`:
        Creates universal logger.
    :callback:`pycnfg.find_path`:
        Finds start script directory and name.

    """
    logger_name = 'logger'
    if not project_path:
        project_path, script_name = find_path()
        logger_name = script_name
    if not logger:
        logger = pycnfg.GetLogger(project_path, logger_name).logger

    # get params from conf.py
    handler = pycnfg.Handler(project_path=project_path, logger=logger)
    configs = handler.read(conf=conf, default_conf=default_conf)
    objects = handler.exec(configs)
    return objects


if __name__ == '__main__':
    pass
