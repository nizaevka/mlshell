"""
The :mod:`sklearn.blocks.model_selection.split` includes utils to split
datasets.
"""

from abc import ABC

import numpy as np
import sklearn

__all__ = ['CustomSplitter']


class CustomSplitter(sklearn.model_selection.BaseCrossValidator, ABC):
    """Custom CV template.

    Attributes
    ----------
    n_splits : int, optional (default=5)
        Number of splits. Must be at least 2.

    """
    def __init__(self, n_splits=5):
        super().__init__()
        self.n_splits = n_splits
        raise NotImplementedError

    def split(self, x, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        groups : array-like of shape (n_samples,), optional (default=None)
            Always ignored, exists for compatibility.

        Yield
        -----
        train : :class:`numpy.ndarray`
            The training set indices for that split.

        test : :class:`numpy.ndarray`
            The testing set indices for that split.

        """
        # Example.
        n = x.shape[0]
        i = 1
        while i <= self.n_splits:
            idx = np.arange(n * (i - 1) / self.n_splits, n * i / self.n_splits,
                            dtype=int)
            yield idx, idx
            i += 1


if __name__ == '__main__':
    pass
