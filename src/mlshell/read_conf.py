"""Module contains class to read configuration file from project directory."""


import importlib.util
import sys
import logging


class GetParams(object):
    """Class to read workflow configuration from file"""
    def __init__(self, logger=None):
        if logger is None:
            self.logger = logging.Logger('GetParams')
        else:
            self.logger = logger
        self.logger = logger

    def get_params(self, project_path):
        self.logger.info("\u25CF READ CONFIGURATION")
        dir_name = project_path.split('/')[-1]
        spec = importlib.util.spec_from_file_location(f'conf', f"{project_path}/conf.py")
        conf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(conf)
        sys.modules['conf'] = conf  # otherwise problem with pickle, depends on the module path
        params = conf.params
        return params

