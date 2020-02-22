"""?"""


from mlshell.libs import get_ipython, os, sys


def find_path():
    """Get fullpath and name of main script

    Returns:
        (tuple): tuple containing:

        - fullpath (str): full path to main script
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

    work_dir = os.getcwd().replace('\\', '/')
    fullpath = work_dir
    # deprecated: always start from inside of project dir
    # check if we are in project_dir()
    # if project_dir is None or project_dir in work_dir:
    #     fullpath = work_dir
    # else:
    #     fullpath='{}/{}'.format(work_dir, project_dir)

    # check if ipython
    if is_ipython():
        script_name = 'ipython'
    else:
        # sys_args provide script_name but not work in Ipython
        temp = sys.argv  # for example ['path/run.py', '55']
        script_name = temp[0].split('/')[-1][:-3]  # run

    return fullpath, script_name