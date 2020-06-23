"""The module contains default configuration."""

import logging
import mlshell.pycnfg as pycnfg

__all__ = ['DEFAULT']


DEFAULT = {
    'path': {
        'default': {
            'init': pycnfg.find_path,
            'class': pycnfg.Producer,
            'global': {},
            'patch': {},
            'priority': 1,
            'steps': [],
        },
    },
    'logger': {
        'default': {
            'init': logging.getLogger('default'),
            'class': pycnfg.Producer,
            'global': {},
            'patch': {},
            'priority': 1,
            'steps': [],
        },
    },

}
"""Default configuration."""


if __name__ == '__main__':
    pass
