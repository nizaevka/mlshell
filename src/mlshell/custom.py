"""
The :mod:`mlshell.custom` contains custom sklearn-based classes.

Notes
-----
All estimators/transformers should specify all ``__init__`` argument explicitly
(no ``*args`` or ``**kwargs``), otherwise grid search not supports.

"""


from abc import ABC

import numpy as np
import sklearn

__all__ = ['FunctionTransformer', 'PowerTransformer', 'OneHotEncoder',
           'CustomSelector', 'CustomReducer', 'PredictionTransformer',
           'MockEstimator', 'ThresholdClassifier', 'CustomCV',
           'cross_val_predict', 'partial_cross_val_predict']


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


class CustomSelector(sklearn.base.BaseEstimator):
    """Pseudo-estimator to scored column for selectors."""

    def __init__(self, estimator_type='classifier', verbose=True, skip=False):
        self.skip = skip
        self.verbose = verbose
        self.feature_importances_ = None
        self.estimator_type = estimator_type
        super().__init__()
        if not self.skip:
            raise NotImplementedError

    def fit(self, x, y):
        if self.skip:
            self.feature_importances_ = np.full(x.shape[1], fill_value=1)
            return self
        # TODO: some logic
        self.feature_importances_ = np.full(x.shape[1], fill_value=1)
        return self


class CustomReducer(sklearn.base.BaseEstimator,
                    sklearn.base.TransformerMixin):
    """Custom dimension reducer template."""

    def __init__(self, skip=False):
        self.skip = skip
        if not skip:
            raise NotImplementedError

    def fit(self, x, y=None):
        if self.skip:
            return self
        # TODO: unsupervised step to analyse/reduce dimension.

        # random_projection
        # sklearn.random_projection.johnson_lindenstrauss_min_dim

        # cluster.FeatureAgglomeration
        # >> import numpy as np
        # >> from sklearn import datasets, cluster
        # >> digits = datasets.load_digits()
        # >> images = digits.images
        # >> x = np.reshape(images, (len(images), -1))
        # >> agglo = cluster.FeatureAgglomeration(n_clusters=32)
        # >> agglo.fit(x)
        # FeatureAgglomeration(affinity='euclidean', compute_full_tree='auto',
        #                      connectivity=None, linkage='ward', memory=None,
        #                      n_clusters=32, pooling_func=...)
        # >> x_reduced = agglo.transform(x)
        # >> x_reduced.shape
        # (1797, 32)

        # reductor = sklearn.decomposition.PCA()
        return self

    def transform(self, x):
        if self.skip:
            return x
        x_transformed = None
        return x_transformed


class PredictionTransformer(sklearn.base.BaseEstimator,
                            sklearn.base.TransformerMixin,
                            sklearn.base.MetaEstimatorMixin):
    """Transformer calls predict_proba on features."""

    def __init__(self, classifier):
        """Replaces features with `clf.predict_proba(x)`"""
        self.clf = classifier

    def fit(self, x, y, **fit_params):
        self.clf.fit(x, y, **fit_params)
        return self

    def transform(self, x):
        return self.clf.predict_proba(x)


class MockEstimator(sklearn.base.BaseEstimator):
    """Estimator always predict input features."""
    def __init__(self):
        pass

    def fit(self, x, y, **fit_params):
        return self

    def predict(self, x):
        return x


class ThresholdClassifier(sklearn.base.BaseEstimator,
                          sklearn.base.ClassifierMixin):
    """Estimator to apply classification threshold.

    Classify samples based on whether they are above of below `threshold`.
    Awaits for predict_proba for features.

    Parameters
    ----------
    threshold : float [0,1], list of float [0,1], None, optional(default=None)
        Classification threshold. For multi-output target list of [n_outputs]
        awaited. If None, np.argmax, that is in binary case equivalent to 0.5.
        If positive class probability P(pos_label) = 1 - P(neg_labels) > th_
        for some sample, classifier predict pos_label for this sample, else
        next label in neg_labels with max probability.

    **kwargs : dict
        kwarg-layer need to set multiple params together in resolver/optimizer.
        {
        'classes': list of np.ndarray
            List of sorted unique labels for each target(s) (n_outputs,
            n_classes).
        'pos_labels': list
            List of "positive" label(s) for target(s) (n_outputs,).
        'pos_labels_ind': list
            List of "positive" label(s) index in np.unique(target) for
            target(s) (n_outputs).
        }

    """
    def __init__(self, threshold=None, **kwargs):
        self.classes = kwargs['classes']
        self.pos_labels = kwargs['pos_labels']
        self.pos_labels_ind = kwargs['pos_labels_ind']
        self.threshold = threshold
        if any(not isinstance(i, np.ndarray) for i in self.classes):
            raise ValueError("Each target 'classes' should be numpy.ndarray.")

    def fit(self, x, y, **fit_params):
        return self

    def predict(self, x):
        # x - predict_proba.
        if not isinstance(x, list):
            # Compliance to multi-output.
            x = [x]
            threshold = [self.threshold]
        else:
            threshold = self.threshold
        assert len(x) == len(self.classes), "Multi-output inconsistent."

        res = []
        for i in range(len(x)):
            # Check that each train fold contain all classes, otherwise can`t
            # predict_proba, since can`t reconstruct probabilities (add zero),
            # cause don`t know which one class is absent.
            n_classes = x[i].shape[1]
            if n_classes != self.classes[i].shape[0]:
                raise ValueError("Train folds missed some classes:\n"
                                 "    ThresholdClassifier "
                                 "can`t identify class probabilities.")

            if threshold[i] is None:
                # Take the one with max probability.
                res.append(self.classes[i].take(np.argmax(x[i],
                                                          axis=1), axis=0))
            else:
                if n_classes > 2:
                    # Multi-class classification.
                    # Remain label with max prob.
                    mask = np.arange(n_classes) != self.pos_labels_ind
                    remain_x = x[i][..., mask]
                    remain_classes = self.classes[i][mask]
                    neg_labels = remain_classes.take(np.argmax(remain_x,
                                                               axis=1), axis=0)
                else:
                    # Binary classification.
                    mask = np.arange(n_classes) != self.pos_labels_ind
                    neg_labels = self.classes[i][mask]
                res.append(
                    np.where(x[i][..., self.pos_labels_ind] > threshold[i],
                             [self.pos_labels], neg_labels)
                )
        return res if len(res) > 1 else res[0]

    def predict_proba(self, x):
        return x


class CustomCV(sklearn.model_selection.BaseCrossValidator, ABC):
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
        train : numpy.ndarray
            The training set indices for that split.

        test : numpy.ndarray
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


def cross_val_predict(*args, **kwargs):
    """Extended sklearn.model_selection.cross_val_predict.

    Add TimeSplitter support (when no first fold prediction).

    Parameters
    ----------
    *args : list
        Passed to sklearn.model_selection.cross_val_predict.
    **kwargs : dict
        Passed to sklearn.model_selection.cross_val_predict.

    Returns
    -------
    y_pred_oof : 2d numpy.ndarray or list of 2d numpy.ndarray
        OOF probability predictions of shape [n_test_samples, n_classes]
        or [n_outputs, n_test_samples, n_classes] for multi-output.
    index_oof : numpy.ndarray or list of numpy.ndarray
        Samples reset indices where predictions available of shape
        [n_test_samples,] or [n_test_samples, n_outputs] for multi-output.

    """
    # Debug key (compare predictions for no TimeSplitter cv strategy).
    _debug = kwargs.pop('_debug', False)
    temp_pp = None
    temp_ind = None
    x = args[1]
    try:
        y_pred_oof = sklearn.model_selection.cross_val_predict(
            *args, **kwargs)
        index_oof = np.arange(0, x.shape[0])
        # [deprecated] y could be None in common case
        # y_index_oof = np.arange(0, y_pred_oof.shape[0])
        if _debug:
            temp_pp = y_pred_oof
            temp_ind = index_oof
            raise ValueError('debug')
    except ValueError:
        y_pred_oof, index_oof = partial_cross_val_predict(*args, **kwargs)
    if _debug:
        assert np.array_equal(temp_pp, y_pred_oof)
        assert np.array_equal(temp_ind, index_oof)
    return y_pred_oof, index_oof


def partial_cross_val_predict(estimator, x, y, cv, fit_params=None,
                              method='predict_proba', **kwargs):
    """Extension to cross_val_predict for TimeSplitter."""
    if fit_params is None:
        fit_params = {}
    if method is not 'predict_proba':
        raise ValueError("Currently only 'predict_proba' method supported.")
    if y.ndim == 1:
        # Add dimension, for compliance to multi-output.
        y = y[..., None]

    def single_output(x_, y_):
        """Single output target."""
        y_pred_oof_ = []
        index_oof_ = []
        ind = 0
        for fold_train_index, fold_test_index in cv.split(x_):
            if hasattr(x_, 'loc'):
                estimator.fit(x_.loc[x_.index[fold_train_index]],
                              y_.loc[y_.index[fold_train_index]],
                              **fit_params)
                # In order of pipeline.classes_.
                fold_y_pred = estimator.predict_proba(
                    x_.loc[x_.index[fold_test_index]])
            else:
                estimator.fit(x_[fold_train_index], y_[fold_train_index],
                              **fit_params)
                # In order of pipeline.classes_.
                fold_y_pred = estimator.predict_proba(x_[fold_test_index])
            index_oof_.extend(fold_test_index)
            y_pred_oof_.extend(fold_y_pred)
            ind += 1
        y_pred_oof_ = np.array(y_pred_oof_)
        index_oof_ = np.array(index_oof_)
        return y_pred_oof_, index_oof_

    # Process targets separately.
    y_pred_oof = []
    index_oof = []
    for i in range(len(y)):
        l, r = single_output(x, y[:, i])
        y_pred_oof.append(l)
        index_oof.append(r)
    index_oof = np.array(index_oof).T
    return y_pred_oof, index_oof


if __name__ == '__main__':
    pass
