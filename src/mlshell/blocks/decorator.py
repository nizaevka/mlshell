"""
The :mod:`mlshell.decorator` includes auxiliary decorators.
"""


import functools

__all__ = ['checker']


def checker(func, options=None):
    """Checks for producers methods.

    Print:

    * Alteration in objects hash.
    * Numpy errors.

     """
    if options is None:
        options = []

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        self = args[0]
        if hasattr(self, 'logger'):
            pfunc = self.logger.info
        else:
            pfunc = print
        hash_before = {key: hash(val) for key, val in self.object.items()}
        result = func(*args, **kwargs)
        hash_after = {key: hash(val) for key, val in self.object.items()}
        hash_diff = {key: {'before': hash_before[key],
                           'after': hash_after[key]}
                     for key in hash_before
                     if hash_before[key] != hash_after[key]}
        if hash_diff:
            pfunc(f"Object(s) hash changed:\n"
                  f"    {hash_diff}")
        if hasattr(self, '_np_error_stat'):
            pfunc(f"Numpy error(s) occurs:\n"
                  f"    {self._np_error_stat}")
            self._np_error_stat = {}
        return result
    return wrapper
