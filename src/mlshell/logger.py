"""The :mod:`mlshell.logger_class` module includes CreateLogger class

TODO: logbook (better developed library) some chamges
"""

from mlshell.libs import os
from mlshell.libs import sys
from mlshell.libs import copy
import logging.config


_dict_log_config = {
    "version": 1,
    "handlers": {
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
            "level": "INFO",
        },
        "warning_handler": {
            "class": "logging.FileHandler",
            "formatter": "json_message",
            "filename": 'warning.log',
            "level": "WARNING",
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
            "stream": None,  # default in std.err (marked),a fter generation will set to std.out
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
            "handlers": ["debug_handler", "info_handler", "warning_handler",
                         "error_handler", "critical_handler", "console_handler", "http_handler"],
            "level": "DEBUG",  # => use handlers level
        },
    },
    "formatters": {
        "json_message": {
            # "()": "pythonjsonlogger.jsonlogger.JsonFormatter",  # pip install pythonjsonlogger
            "format": "%(levelname)s : %(message)s",
        },
        "message": {
            "format": "%(message)s %(delimeter)s"
        }
    }
}
"""(dict): logger configuration for logging.config.dictConfig method"""


class CreateLogger(object):
    """Create logger object."""

    def __init__(self, fullpath, **kwargs):
        """Create logger and files.

        use _dict_log_config for logger config
        create dir from fullpath if not exist
        create logger with name = last fullpath dir
        create logs files "(logger_name)_(level).log"

        Args:
            fullpath (str): path to dir for logs files
            url (str): if not None used to send logs over http_handler from dictLogConfig.
                (default: None)
            host (str): if url is not None use for http_handler.
            logger_name(str): used in logs files
                (default: last fullpath dir)
        """
        self.logger = None
        self.__call__(fullpath, **kwargs)

    def __call__(self, fullpath, logger_name=None, url=None, host=None):
        # create dir
        if not os.path.exists(fullpath):
            os.makedirs(fullpath)
        # get logger name from fullpath
        if logger_name:
            logger_name = fullpath[:-1].split('/')[-1]
        # update path in config
        _dict_log_config["loggers"][logger_name] = copy.deepcopy(_dict_log_config["loggers"]["mylogger"])
        _dict_log_config["handlers"]["debug_handler"]["filename"] = '{}/{}_debug.log'.format(fullpath, logger_name)
        _dict_log_config["handlers"]["info_handler"]["filename"] = '{}/{}_info.log'.format(fullpath, logger_name)
        _dict_log_config["handlers"]["warning_handler"]["filename"] = '{}/{}_warning.log'.format(fullpath, logger_name)
        _dict_log_config["handlers"]["error_handler"]["filename"] = '{}/{}_error.log'.format(fullpath, logger_name)
        _dict_log_config["handlers"]["critical_handler"]["filename"] = '{}/{}_critical.log'.format(fullpath,
                                                                                                   logger_name)
        _dict_log_config["handlers"]["console_handler"]["stream"] = sys.stdout
        if not url:
            # del http-handler
            del _dict_log_config["loggers"][logger_name]['handlers']['http_handler']
        else:
            _dict_log_config["loggers"][logger_name]['handlers']['http_handler']['url'] = url
            _dict_log_config["loggers"][logger_name]['handlers']['http_handler']['host'] = host

        # create logger object (auto generate files)
        logging.config.dictConfig(_dict_log_config)
        logger = logging.getLogger(logger_name)
        # add extra in every entry
        extra = {'delimeter': '\n' + '+' * 100}
        self.logger = logging.LoggerAdapter(logger, extra)
        # clean debug and warning files
        with open(_dict_log_config["handlers"]["debug_handler"]["filename"], 'w'):
            with open(_dict_log_config["handlers"]["warning_handler"]["filename"], 'w'):
                pass


if __name__ == 'main':
    pass
