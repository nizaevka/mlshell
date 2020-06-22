"""The module contains default configuration."""


import mlshell.pycnfg as pycnfg

__all__ = ['DEFAULT']


DEFAULT = {
    'section': {
        'configuration': {
            'init': {},
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
