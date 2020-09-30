"""The :mod:`mlshell.producers.logger` includes utils to ctreate logger

:class:`mlshell.LoggerProducer`
:data:`mlshell.LoggerConfig`
"""


import copy
import logging.config
import os
import sys
import time

import pycnfg

__all__ = ['LoggerProducer', 'LOGGER_CONFIG']


class LevelFilter(object):
    """Custom filter."""
    def __init__(self, level=50):
        self._level = level

    def filter(self, record):
        if record.levelno <= self._level:
            return True
        else:
            return False


class CustomFormatter(logging.Formatter):
    def format(self, record):
        s = super().format(record)
        s = s.replace('    |__ ', '     \u25B6 ')
        s = s.replace('|__ ', '\u25CF ')
        return s


LOGGER_CONFIG = {
    "version": 1,
    'filters': {
        'test_level_filter': {
            '()': LevelFilter,
            'level': 5,
        }
    },
    "handlers": {
        "test_handler": {
            "class": "logging.FileHandler",
            "formatter": "custom",
            "filename": 'test.log',
            "level": 5,
            "filters": ['test_level_filter']
        },
        "debug_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "maxBytes": 25000000,  # 25mb 25*10^6 byte.
            "backupCount": 3,      # Amount of backup files.
            "formatter": "custom",
            "filename": 'debug.log',  # Init only, will be replaced.
            "level": "DEBUG",
        },
        "info_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "maxBytes": 25000000,
            "backupCount": 5,
            "formatter": "custom",
            "filename": 'info.log',
            "level": "INFO",  # 20
        },
        "minimal_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "maxBytes": 25000000,
            "backupCount": 5,
            "formatter": "custom",
            "filename": 'info.log',
            "level": 25,
        },
        "warning_handler": {
            "class": "logging.FileHandler",
            "formatter": "default",
            "filename": 'warning.log',
            "level": "WARNING",  # 30
        },
        "error_handler": {
            "class": "logging.FileHandler",
            "formatter": "custom",
            "filename": 'error.log',
            "level": "ERROR",
        },
        "critical_handler": {
            "class": "logging.FileHandler",
            "formatter": "custom",
            "filename": 'critical.log',
            "level": "CRITICAL",
        },
        "console_handler": {
            "class": "logging.StreamHandler",
            "formatter": "custom",
            "level": "INFO",
            "stream": None,  # Default in std.err (marked).
        },
        "http_handler": {
            "class": "logging.handlers.HTTPHandler",
            "formatter": "custom",
            "level": "ERROR",
            "host": 'www.example.com',
            "url": 'https://wwww.example.com/address',
            "method": "POST",
        },
    },
    "loggers": {
        "default": {
            "handlers": ["test_handler", "debug_handler", "info_handler",
                         "minimal_handler", "warning_handler", "error_handler",
                         "critical_handler", "console_handler", "http_handler"
                         ],
            "level": 1,  # => use handlers levels.
        },
    },
    "formatters": {
        "default": {
            "format": "%(levelname)s: %(message)s%(delimeter)s"
        },
        "custom": {
            "()": CustomFormatter,
            "format": "%(message)s%(delimeter)s",
        },
    }
}
"""dict : Logger configuration for logging.config.dictConfig method.

Levels of logging:

* critical
    reset on logger creation.
* error
    reset on logger creation.
* warning
    reset on logger creation.
* minimal
    cumulative.
    only score for best run in gs.
* info
    cumulative.
    workflow information.
* debug
    reset on logger creation.
    detailed workflow information.
* test
    only for test purposes (will be created only in pytest env).
        
"""


class LoggerProducer(pycnfg.Producer):
    """Factory to produce logger.

    Interface: make.

    Parameters
    ----------
    objects : dict
        Dictionary with objects from previous executed producers:
        {'section_id__config__id', object,}.
    oid : str
        Unique identifier of produced object.
    path_id : str, optional (default='default')
        Project path identifier in `objects`.

    Attributes
    ----------
    objects : dict
        Dictionary with objects from previous executed producers:
        {'section_id__config__id', object,}.
    oid : str
        Unique identifier of produced object.
    project_path: str
        Absolute path to project dir.

    """
    _required_parameters = ['objects', 'oid', 'path_id']

    def __init__(self, objects, oid, path_id='path__default'):
        super().__init__(objects, oid, path_id=path_id)

    def make(self, logger_name, fullpath=None, config=None,
             extra=None, clean=None, **kwargs):
        """Create logger object and corresponding file descriptors.

        Parameters
        ----------
        logger_name : str
            Logger identifier in ``config`` .
        fullpath : str, optional (default=None)
            Absolute path to dir for logs files or relative to
            'project_dir' started with './'. Created, if not exists.
            If None, used ``project_path/results`` .
        config : dict, optional (default=None)
            Logger configuration to pass in :func:`logging.config.dictConfig` .
            If None, :data:`mlshell.LOGGER_CONFIG` is used.
        extra : dict, optional (default=None)
            Add extra to :class:`logging.LoggerAdapter` . If None:
            ``{'delimiter': '='*79}``.
        clean : list of str, optional (default=None)
            List of handlers identifiers, for which clean corresponding files.
            If None, ["debug_handler", "warning_handler", "error_handler",
            "critical_handler"].
        **kwargs : dict
            User-defined params for handlers, updating `config`.
            For example: {'http_hadler':
            {
            'host':'www.example.com',
            'url':'https://wwww.example.com/address'
            }}

        Notes
        -----
        Logs files named "(logger_name)_(level).log".

        ``test_handler`` used only if program run with pytest.

        ``http_handler`` used only if configuration provided in kwargs.

        """
        if fullpath is None:
            fullpath = f"{self.project_path}/results/logs_{logger_name}"
        elif fullpath.startswith('./'):
            fullpath = f"{self.project_path}/{fullpath[2:]}"
        if config is None:
            config = copy.deepcopy(LOGGER_CONFIG)
        if extra is None:
            # Add extra in every entry.
            extra = {'delimeter': '\n' + '=' * 79}
        if clean is None:
            clean = ["debug_handler", "warning_handler",
                     "error_handler", "critical_handler"]

        # Create dir.
        if not os.path.exists(fullpath):
            os.makedirs(fullpath)
        if logger_name not in config["loggers"]:
            raise KeyError(f"Unknown logger name {logger_name}")
        # Update path for config/params in handlers.
        handlers = set()
        for handler_name in LOGGER_CONFIG["handlers"]:
            prefix = handler_name.split('_')[0]
            if prefix in ['test', 'debug', 'info', 'minimal',
                          'warning', 'error', 'critical']:
                config["handlers"][handler_name]['filename'] =\
                    f"{fullpath}/{logger_name}_{prefix}.log"
            if prefix is 'console':
                config["handlers"][handler_name]['stream'] = sys.stdout
            dic = kwargs.get(handler_name, {})
            config['handlers'][handler_name].update(dic)
            handlers.add(handler_name)
        # Special cases.
        if "PYTEST_CURRENT_TEST" not in os.environ:
            # Delete test handler.
            self._del_handler(config, logger_name, handlers, 'test_handler')
        if 'http_handler' not in kwargs:
            # Delete http handler.
            self._del_handler(config, logger_name, handlers, 'http_handler')
        # Otherwise logs mixing.
        sys.stdout.flush()
        time.sleep(0.1)
        # Create logger object (auto generate files).
        logging.config.dictConfig(config)
        logger = logging.getLogger(logger_name)
        logger = logging.LoggerAdapter(logger, extra)
        # Clean file(s).
        for hadler in clean:
            with open(config["handlers"][hadler]["filename"], 'w'):
                pass
        logger.log(25, '<>' * 40)
        return logger

    def _del_handler(self, config, logger_name, handlers, handler_name):
        if handler_name in handlers:
            index = config["loggers"][logger_name]['handlers']\
                .index(handler_name)
            del config["loggers"][logger_name]['handlers'][index]


if __name__ == '__main__':
    pass
