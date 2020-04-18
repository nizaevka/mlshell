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


def dic_flatter(dic, dic_flat, key_transform=None, val_transform=None, keys_lis_prev=None, max_depth=None):
    """Flatten the dict.

    {'a':{'b':[], 'c':[]},} =>  {('a','b'):[], ('a','c'):[]}

    Args:
        dic (dict): input dictionary.
        dic_flat (dict): result dictionary.
        key_transform (callback, optional (default=None)): apply to end keys,
            if None '__'.join() is used.
        val_transform (callback, optional (default=None)): apply to end dic values,
            if None identity function f(x)=x is used.
        keys_lis_prev: need for recursion.
        max_depth (None or int, optional (default=None)): depth of recursion.

    Note:
        if dict is empty, stop.

    """
    if val_transform is None:
        def id_func(x): return x
        val_transform = id_func
    if key_transform is None:
        key_transform = tuple

    if keys_lis_prev is None:
        keys_lis_prev = []
    for key, val in dic.items():
        keys_lis = keys_lis_prev[:]
        keys_lis.append(key)
        if isinstance(val, dict) and val and (len(keys_lis_prev) < max_depth or not max_depth):
            dic_flatter(val, dic_flat,
                        key_transform=key_transform, val_transform=val_transform,
                        keys_lis_prev=keys_lis, max_depth=max_depth)
        else:
            if len(keys_lis) == 1:
                dic_flat[keys_lis[0]] = val_transform(val)
            else:
                dic_flat[key_transform(keys_lis)] = val_transform(val)


def json_keys2int(x):
    # dic key to int
    if isinstance(x, dict) and 'categoric' not in x:
        return {int(k): v for k, v in x.items()}
    return x


if __name__ == '__main__':
    pass
