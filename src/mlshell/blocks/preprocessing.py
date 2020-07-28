"""
The :mod:`mlshell.preprocessing` includes skip-able sklearn transformers.
"""


import sklearn
import numpy as np

__all__ = ['FunctionTransformer', 'PowerTransformer', 'OneHotEncoder']


class FunctionTransformer(sklearn.preprocessing.FunctionTransformer):
    """Extended FunctionTransformer.

     Skip argument added.

     """
    def __init__(self, func=None, inverse_func=None, validate=False,
                 accept_sparse=False, check_inverse=True, kw_args=None,
                 inv_kw_args=None, skip=False):
        self.skip = skip
        super().__init__(func=func, inverse_func=inverse_func,
                         validate=validate, accept_sparse=accept_sparse,
                         check_inverse=check_inverse, kw_args=kw_args,
                         inv_kw_args=inv_kw_args)

    def fit(self, x, y=None):
        if self.skip:
            return self
        else:
            return super().fit(x, y)

    def fit_transform(self, x, y=None, **fit_params):
        if self.skip:
            return x
        else:
            return super().fit_transform(x, y, **fit_params)

    def transform(self, x):
        if self.skip:
            return x
        else:
            return super().transform(x)

    def inverse_transform(self, x):
        if self.skip:
            return x
        else:
            return super().inverse_transform(x)


class PowerTransformer(sklearn.preprocessing.PowerTransformer):
    """Extended PowerTransformer.

     Skip argument added.

     """
    def __init__(self, method='yeo-johnson', standardize=True, copy=True,
                 skip=False):
        self.skip = skip
        super().__init__(method=method, standardize=standardize, copy=copy)

    def fit(self, x, y=None):
        if self.skip:
            return self
        else:
            return super().fit(x, y)

    def fit_transform(self, x, y=None):
        if self.skip:
            return x
        else:
            return super().fit_transform(x, y)

    def transform(self, x):
        if self.skip:
            return x
        else:
            return super().transform(x)

    def inverse_transform(self, x):
        if self.skip:
            return x
        else:
            return super().inverse_transform(x)


class OneHotEncoder(sklearn.preprocessing.OneHotEncoder):
    """Extended OneHotEncoder.

    Skip argument added. Also if x=[], trigger skip.

    """
    def __init__(self, categories=None, drop=None, sparse=True,
                 dtype=np.float64, handle_unknown='error', skip=False):
        self.skip = skip
        super().__init__(categories=categories, drop=drop, sparse=sparse,
                         dtype=dtype, handle_unknown=handle_unknown)

    def fit(self, x, y=None):
        self._check_empty(x)
        if self.skip:
            return self
        else:
            return super().fit(x, y)

    def fit_transform(self, x, y=None):
        self._check_empty(x)
        if self.skip:
            return x
        else:
            return super().fit_transform(x, y)

    def transform(self, x):
        self._check_empty(x)
        if self.skip:
            return x
        else:
            return super().transform(x)

    def _check_empty(self, x):
        if x.size == 0:
            self.skip = True

    def inverse_transform(self, x):
        if self.skip:
            return x
        else:
            return super().inverse_transform(x)


if __name__ == '__main__':
    pass
