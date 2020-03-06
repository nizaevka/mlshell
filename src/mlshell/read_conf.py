"""Module contains class to read configuration file from project directory."""


import importlib.util
import sys


class GetParams(object):
    """Class to read workflow configuration from file"""
    def __init__(self, project_path):
        dir_name = project_path.split('/')[-1]
        spec = importlib.util.spec_from_file_location(f'conf', f"{project_path}/conf.py")
        conf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(conf)
        sys.modules['conf'] = conf  # otherwise problem with pickle, depends on the module path
        self.params = conf.params
