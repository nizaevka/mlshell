"""The :mod:`pycnfg.logger` module includes LoggerProducer class."""


import copy
import logging.config
import os
import sys

import mlshell.pycnfg as pycnfg


class LevelFilter(object):
    """Custom filter for logger configuration."""
    def __init__(self, level=50):
        self._level = level

    def filter(self, record):
        if record.levelno <= self._level:
            return True
        else:
            return False


class CustomFormatter(logging.Formatter):
    """Custom formatter for logger configuration."""
    def format(self, record):
        record.message.replace('|__ ', '\u25CF ')
        record.message.replace('    |__ ', '\u25CF \u25B6 ')
        return record


CONFIG = {
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
            "formatter": "default",
            "filename": 'test.log',
            "level": 5,
            "filters": ['test_level_filter']
        },
        "debug_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "maxBytes": 25000000,  # 25mb 25*10^6 byte.
            "backupCount": 3,      # Amount of backup files.
            "formatter": "default",
            "filename": 'debug.log',  # Init only, will be replaced.
            "level": "DEBUG",
        },
        "info_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "maxBytes": 25000000,
            "backupCount": 5,
            "formatter": "default",
            "filename": 'info.log',
            "level": "INFO",  # 20
        },
        "minimal_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "maxBytes": 25000000,
            "backupCount": 5,
            "formatter": "default",
            "filename": 'info.log',
            "level": 25,
        },
        "warning_handler": {
            "class": "logging.FileHandler",
            "formatter": "custom",
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
            "formatter": "default",
            "level": "INFO",
            "stream": None,  # Default in std.err (marked).
        },
        "http_handler": {
            "class": "logging.handlers.HTTPHandler",
            "formatter": "default",  # Ignore custom (as use LogRecord object).
            "level": "ERROR",
            "host": 'www.example.com',
            "url": 'https://wwww.example.com/address',
            "method": "POST",
        },
    },
    "loggers": {
        "logger": {
            "handlers": ["test_handler", "debug_handler", "info_handler",
                         "minimal_handler", "warning_handler", "error_handler",
                         "critical_handler", "console_handler", "http_handler"],
            "level": 1,  # => use handlers levels.
        },
    },
    "formatters": {
        "default": {
            "format": "%(message)s%(delimeter)s"
        }
        "custom": {
            "()": CustomFormatter,
            "format": "%(message)s%(delimeter)s",  # %(levelname)s:
        },
    }
}
"""(dict): Logger configuration for logging.config.dictConfig method."""


class LoggerProducer(pycnfg.Producer):
    """Create logger object."""

    def __init__(self, project_path):
        self.project_path = project_path
        TODO:

    def create(self, logger, fullpath=None, logger_name='logger',
                config=None, extra=None, clean=None, **kwargs):
        """Create logger object and corresponding file descriptors.

        Parameters
        ----------
        logger : object with logging.Logger interface
            For compliance with pycnfg configuration (ignored).
        fullpath : str, optional (default=None)
            Absolute path to dir for logs files. Created, if not exists.
            If None, used "project_path/results/logs_{logger_name}".
        logger_name: str, optional (default='logger')
            Logger identifier.
        config : dict, optional (default=None)
            Logger configuration to pass in logging.config.dictConfig. If None,
            module level `CONFIG` is used.
        extra : dict, optional (default=None)
            Add extra to logging.LoggerAdapter. If None, {'delimeter': '\n' +
            '=' * 100}.
        clean : list of str, optional (default=None)
            List of handlers identifiers, for which clean corresponding files.
            If None, ["debug_handler", "warning_handler", "error_handler",
            "critical_handler"].
        **kwargs : dict
            User-defined params for handlers, will update `config`.
            For example:
            {'http_hadler':{'host':'www.example.com',
                            'url':'https://wwww.example.com/address'}}

        Notes
        -----
        Logs files created as "(logger_name)_(level).log".
        `test_handler` used only if program run with pytest.
        `http_hanfler` used only if configuration provided in kwargs.

        """
        if not fullpath:
            fullpath = f"{self.project_path}/results/logs_{logger_name}"
        if not config:
            config = copy.deepcopy(CONFIG)
        if not extra:
            # Add extra in every entry.
            extra = {'delimeter': '\n' + '=' * 100}
        if not clean:
            clean = ["debug_handler", "warning_handler",
                     "error_handler", "critical_handler"]

        # Create dir.
        if not os.path.exists(fullpath):
            os.makedirs(fullpath)
        if logger_name != 'logger':
            config["loggers"][logger_name] = config["loggers"].pop("logger")
        # Update path for config/params in handlers.
        handlers = set()
        for handler_name in CONFIG["handlers"]:
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
        # Create logger object (auto generate files).
        logging.config.dictConfig(config)
        logger = logging.getLogger(logger_name)
        logger = logging.LoggerAdapter(logger, extra)
        # Clean file(s).
        for hadler in clean:
            with open(config["handlers"][hadler]["filename"], 'w'):
                pass
        logger.log(25, '\n' + '<>' * 50)
        return logger

    def _del_handler(self, config, logger_name, handlers, handler_name):
        if handler_name in handlers:
            index = config["loggers"][logger_name]['handlers']\
                .index(handler_name)
            del config["loggers"][logger_name]['handlers'][index]


if __name__ == '__main__':
    pass
