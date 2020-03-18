"""The module with auxiliary functions"""


from mlshell.libs import get_ipython, os, sys


def find_path(filepath=None):
    """Get fullpath and name of main script

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
                return True  # Jupyter notebook or qtconsole
            elif shell == 'TerminalInteractiveShell':
                return True  # Terminal running IPython
            else:
                return False  # Other type (?)
        except NameError:
            return False  # Probably standard Python interpreter

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
