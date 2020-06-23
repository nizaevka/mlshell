"""The module includes auxiliary utilities."""


import os
import sys

from IPython import get_ipython
import mlshell.pycnfg as pycnfg

__all__ = ['find_path', 'run']


def find_path(script_name=False, filepath=None):
    """Get fullpath of main script.

    Parameters
    ----------
    script_name : bool, optional (default=False)
        If True, return also script name.
    filepath : str, None, optional (default=None)
        Path to main script. If None and Ipython, get from workdir. If None and
        standard interpreter, get from sys.argv. If sys.argv empty, get from
        working directory.

    Returns
    -------
    project_dir : str
        Full path to start script directory.
    script_name : str, otional if `script_name` is True
        Main script name. 'ipython' for Ipython.

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
            # Probably standard Python interpreter.
            return False
    if filepath is not None:
        temp = filepath.replace('\\', '/').split('/')
        project_dir = '/'.join(temp[:-1])
        script_name_ = temp[-1][:-3]
    elif is_ipython():
        project_dir = os.getcwd().replace('\\', '/')
        script_name_ = 'ipython'
    else:
        # sys.argv provide script_name, but not work in Ipython.
        # For example ['path/run.py', '55'].
        ext_size = 3  # '.py'
        script_name_ = sys.argv[0]\
            .replace('\\', '/')\
            .split('/')[-1][:-ext_size]  # run
        project_dir = sys.argv[0]\
            .replace('\\', '/')[:-len(script_name_)-ext_size]
    if script_name:
        return project_dir, script_name_
    else:
        return project_dir


def run(conf, default_conf=None):
    """Wrapper over configuration handler.

    Parameters
    ----------
    conf : dict
        Configuration to pass in `pycnfg.Handler().read()`.
    default_conf : None, dict, optional (default=None)
        Default configurations to pass in `pycnfg.Handler().read()`.

    Returns
    -------
    objects : dict {'configuration_id': object}.
        Dict of objects created by execution all configurations.

    See Also
    --------
    :class:`pycnfg.Handler`:
        Reads configurations, executes steps.

    """
    handler = pycnfg.Handler()
    configs = handler.read(conf=conf, default_conf=default_conf)
    objects = handler.exec(configs)
    return objects


if __name__ == '__main__':
    pass
