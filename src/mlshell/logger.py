"""The :mod:`mlshell.logger_class` module includes CreateLogger class

TODO: logbook (better developed library)
"""

from mlshell.libs import os
from mlshell.libs import sys
from mlshell.libs import copy
import logging.config


# (dict): logger configuration for logging.config.dictConfig method
_dict_log_config = {
    "version": 1,
    "handlers": {
        "debug_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "maxBytes": 25000000,  # 25mb 25*10^6 byte
            "backupCount": 3,      # amount of backup files
            "formatter": "message",
            "filename": 'debug.log',  # just for initialization, will replcace
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
            "stream": None,  # default in std.err (marked),after generation will set to std.out
        },
        "http_handler": {   # ignore jsonFormatter (cause use LogRecord object)
            "class": "logging.handlers.HTTPHandler",
            "formatter": "message",  # игнорируется
            "level": "ERROR",
            "host": 'nizaevka.pythonanywhere.com',
            "url": 'https://nizaevka.pythonanywhere.com/fe0f200d-c9ab-4a7d-8ec8-cf56d4c08ec2',
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
            # "()": "pythonjsonlogger.jsonlogger.JsonFormatter",  pip install pythonjsonlogger
            "format": "%(levelname)s : %(message)s",
        },
        "message": {
            "format": "%(message)s %(delimeter)s"
        }
    }
}


class CreateLogger(object):
    """Create logger object."""

    def __init__(self, fullpath, send_http=False):
        """Initialize class object."""
        self.logger = None
        self.__call__(fullpath, send_http)

    def __call__(self, fullpath, send_http=False, logger_name=None):
        """Create logger and files.

        use _dict_log_config for logger config
        create dir from fullpath if not exist
        create logger with name = last fullpath dir
        create logs files "(logger_name)_(level).log"

        Args:
            fullpath (str): path to dir for logs files
            send_http (bool): if True send logs over http_handler from dictLogConfig
                (default: False)
            logger_name(str): used in logs files
                (default: last fullpath dir)
        """
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
        if not send_http:
            # del http-handler
            del _dict_log_config["loggers"][logger_name]['handlers'][-1]
        # create logger object (auto generate files)
        logging.config.dictConfig(_dict_log_config)
        logger = logging.getLogger(logger_name)
        # add extra in every entry
        extra = {'delimeter': '\n'+'+' * 100}
        self.logger = logging.LoggerAdapter(logger, extra)
        # clean debug and warning files
        with open(_dict_log_config["handlers"]["debug_handler"]["filename"], 'w'):
            with open(_dict_log_config["handlers"]["warning_handler"]["filename"], 'w'):
                pass


if __name__ == 'main':
    pass
