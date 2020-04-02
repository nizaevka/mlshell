"""The :mod:`mlshell.logger_class` module includes CreateLogger class

TODO: logbook (better developed library) some chamges
"""

from mlshell.libs import os
from mlshell.libs import sys
from mlshell.libs import copy
import logging.config


class LevelFilter(object):
    def __init__(self, level=50):
        self._level = level

    def filter(self, log_record):
        if log_record.levelno <= self._level:
            return True
        else:
            return False


_dict_log_config = {
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
            "formatter": "message",
            "filename": 'test.log',
            "level": 5,
            "filters": ['test_level_filter']
        },
        "debug_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "maxBytes": 25000000,  # 25mb 25*10^6 byte
            "backupCount": 3,      # amount of backup files
            "formatter": "message",
            "filename": 'debug.log',  # only for initialization, will be replaced
            "level": "DEBUG",
        },
        "info_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "maxBytes": 25000000,
            "backupCount": 5,
            "formatter": "message",
            "filename": 'info.log',
            "level": "INFO",  # 20
        },
        "minimal_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "maxBytes": 25000000,
            "backupCount": 5,
            "formatter": "message",
            "filename": 'info.log',
            "level": 25,
        },
        "warning_handler": {
            "class": "logging.FileHandler",
            "formatter": "json_message",
            "filename": 'warning.log',
            "level": "WARNING",  # 30
        },
        "error_handler": {
            "class": "logging.FileHandler",
            "formatter": "json_message",
            "filename": 'error.log',
            "level": "ERROR",
        },
        "critical_handler": {
            "class": "logging.FileHandler",
            "formatter": "json_message",
            "filename": 'critical.log',
            "level": "CRITICAL",
        },
        "console_handler": {
            "class": "logging.StreamHandler",
            "formatter": "message",
            "level": "INFO",
            "stream": None,  # default in std.err (marked)
        },
        "http_handler": {   # ignore jsonFormatter (cause use LogRecord object)
            "class": "logging.handlers.HTTPHandler",
            "formatter": "message",  # ignored
            "level": "ERROR",
            "host": 'www.example.com',
            "url": 'https://wwww.example.com/address',
            "method": "POST",
        },
    },
    "loggers": {
        "mylogger": {
            "handlers": ["test_handler", "debug_handler", "info_handler", "minimal_handler", "warning_handler",
                         "error_handler", "critical_handler", "console_handler", "http_handler"],
            "level": 1,  # => use handlers level
        },
    },
    "formatters": {
        "json_message": {
            # pip install pythonjsonlogger
            # "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": "%(message)s%(delimeter)s",  # %(levelname)s:\n
        },
        "message": {
            "format": "%(message)s%(delimeter)s"
        }
    }
}
"""(dict): logger configuration for logging.config.dictConfig method"""


class CreateLogger(object):
    """Create logger object."""
    def __init__(self, project_path, logger_name, **kwargs):
        """Create logger and files.

        use _dict_log_config for logger config
        create dir from fullpath if not exist
        create logger with name = last fullpath dir
        create logs files "(logger_name)_(level).log"

        Args:
            project_path (str): path to dir for logs files
            kwargs (dict): user-defined params for handlers.
                {'http_hadler':{'host':'www.example.com','url':'https://wwww.example.com/address'}}
            logger_name(str): used in logs files
                (default: last fullpath dir)
        """
        self.logger = self.create_logger(project_path, logger_name, **kwargs)

    def create_logger(self, project_path, logger_name, **kwargs):
        fullpath = f"{project_path}/results/logs_{logger_name}"
        # create dir
        if not os.path.exists(fullpath):
            os.makedirs(fullpath)

        dict_log_config = copy.deepcopy(_dict_log_config)
        dict_log_config["loggers"][logger_name] = dict_log_config["loggers"].pop("mylogger")
        # update path in config/params in handlers
        handlers = set()
        for handler_name in _dict_log_config["handlers"]:
            prefix = handler_name.split('_')[0]
            if prefix in ['test', 'debug', 'info', 'minimal', 'warning', 'error', 'critical']:
                dict_log_config["handlers"][handler_name]['filename'] = '{}/{}_{}.log'.format(fullpath,
                                                                                               logger_name, prefix)
            if prefix is 'console':
                dict_log_config["handlers"][handler_name]['stream'] = sys.stdout
            dic = kwargs.get(handler_name, {})
            dict_log_config['handlers'][handler_name].update(dic)
            handlers.add(handler_name)

        # special cases
        if "PYTEST_CURRENT_TEST" not in os.environ:
            self.del_handler(dict_log_config, logger_name, handlers, 'test_handler')
        if 'http_handler' not in kwargs:
            self.del_handler(dict_log_config, logger_name, handlers, 'http_handler')

        # create logger object (auto generate files)
        logging.config.dictConfig(dict_log_config)
        logger = logging.getLogger(logger_name)
        # add extra in every entry
        extra = {'delimeter': '\n' + '=' * 100}
        logger = logging.LoggerAdapter(logger, extra)
        # clean file(s)
        for hadler in ["debug_handler", "warning_handler", "error_handler", "critical_handler"]:
            with open(dict_log_config["handlers"][hadler]["filename"], 'w'):
                pass
        logger.log(25, '\n' + '<>' * 50)
        return logger

    def del_handler(self, dict_log_config, logger_name, handlers, handler_name):
        if handler_name in handlers:
            index = dict_log_config["loggers"][logger_name]['handlers'].index(handler_name)
            del dict_log_config["loggers"][logger_name]['handlers'][index]


if __name__ == 'main':
    pass
