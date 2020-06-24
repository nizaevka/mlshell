"""The module contains custom classes on base of sklearn classes

Note:
    All estimators/transformers should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``). Otherwise will be problem in GS.

"""


from abc import ABC


from mlshell.libs import *


class SkippablePowerTransformer(sklearn.preprocessing.PowerTransformer):
    """Skippable version of PowerTransformer."""

    def __init__(self, method='yeo-johnson', standardize=True, copy=True, skip=False):
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


class SkippableOneHotEncoder(sklearn.preprocessing.OneHotEncoder):
    """Skippable version of OneHotEncoder."""

    def __init__(self, categories=None, drop=None, sparse=True, dtype=np.float64,
                 handle_unknown='error', skip=False):
        # can work if no categor functions x=[]
        self.skip = skip
        super().__init__(categories=categories, drop=drop, sparse=sparse, dtype=dtype,
                         handle_unknown=handle_unknown)
        # [deprecated] forbade example with kwargs
        # def __init__(self, **kwarg):
        #     params = dict(categories=None, drop=None, sparse=True, dtype=np.float64,
        #              handle_unknown='error', skip=False)
        #     params.update(kwarg)
        #     self.skip=params.pop('skip')
        #     super().__init__(**params)

    def fit(self, x, y=None):
        self.check_empty(x)
        if self.skip:
            return self
        else:
            return super().fit(x, y)

    def fit_transform(self, x, y=None):
        self.check_empty(x)
        if self.skip:
            return x
        else:
            return super().fit_transform(x, y)

    def transform(self, x):
        self.check_empty(x)
        if self.skip:
            return x
        else:
            return super().transform(x)

    def check_empty(self, x):
        if x.size == 0:
            self.skip = True


class CustomReducer(sklearn.base.BaseEstimator):
    """Class custom dimension reducer """

    def __init__(self, skip=False):
        self.skip = skip
        if not skip:
            raise NotImplementedError

    def fit(self, x, y=None):
        if self.skip:
            return self
        # TODO: unsupervised step to reduce dimension or analyse

        # random_projection
        # sklearn.random_projection.johnson_lindenstrauss_min_dim

        # # cluster.FeatureAgglomeration
        # >> > import numpy as np
        # >> > from sklearn import datasets, cluster
        # >> > digits = datasets.load_digits()
        # >> > images = digits.images
        # >> > x = np.reshape(images, (len(images), -1))
        # >> > agglo = cluster.FeatureAgglomeration(n_clusters=32)
        # >> > agglo.fit(x)
        # FeatureAgglomeration(affinity='euclidean', compute_full_tree='auto',
        #                      connectivity=None, linkage='ward', memory=None, n_clusters=32,
        #                      pooling_func=...)
        # >> > x_reduced = agglo.transform(x)
        # >> > x_reduced.shape
        # (1797, 32)

        # reductor = decomposition.PCA()
        return self

    def transform(self, x):
        if self.skip:
            return x
        x_transformed = None
        return x_transformed

    def fit_transform(self, x, y=None):
        if self.skip:
            return x
        else:
            self.fit(x, y).transform(x)


class CustomImputer(object):
    """Class custom imputer."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        sklearn.impute.SimpleImputer()
        raise NotImplementedError
        # replcae np.nan на уникальное + add indicator feature
        # если просто заменить на 0, признак давать вклад в предсказание данного объекта не будет,
        # то потеряем информацию без индикатора
        # для линейных моделей - не должyj быть выбросом, можно сделать регрессию и предсказать
        # для деревьев - просто отнесет в свою ветку
        # надо кастомный класс, в зависимости от типа алгоритма своё ставить


class CustomCV(sklearn.model_selection.BaseCrossValidator, ABC):
    """Custom CV

    Attributes:
        n_splits (int): Number of splits. Must be at least 2. (default=5)

    """

    def __init__(self, n_splits=5):
        super().__init__()
        self.n_splits = n_splits
        raise NotImplementedError

    def split(self, x, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Args:
            x (array-like): shape (n_samples, n_features)
                Training data, where n_samples is the number of samples
                and n_features is the number of features.

            y (array-like): shape (n_samples,)
                Always ignored, exists for compatibility.

            groups (array-like): shape (n_samples,), optional
                Always ignored, exists for compatibility.

        Yields:
            train (ndarray):
                The training set indices for that split.

            test (ndarray):
                The testing set indices for that split.
                
        """
        # [Example]
        n = x.shape[0]
        i = 1
        while i <= self.n_splits:
            idx = np.arange(n * (i - 1) / self.n_splits, n * i / self.n_splits, dtype=int)
            yield idx, idx
            i += 1


class CustomSelectorEstimator(sklearn.base.BaseEstimator):
    """Class pseudo-estimator to scored column for selectors """

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


class PredictionTransformer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin, sklearn.base.MetaEstimatorMixin):
    """Class to add brutforce of th in GridSearch"""

    def __init__(self, classifier):
        """Replaces all features with `clf.predict_proba(x)`"""
        self.clf = classifier

    def fit(self, x, y, **fit_params):
        self.clf.fit(x, y, **fit_params)
        return self

    def transform(self, x):
        return self.clf.predict_proba(x)


# TODO: remain binary simple variant
class ThresholdClassifier(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    """Classify samples based on whether they are above of below `threshold`.

    Parameters
    ----------
    threshold : float [0,1], None, list of float [0,1]
        For multioutput target list of [n_outouts].

    Args:
        classes (array of shape (n_classes), or a list of n_outputs such arrays if n_outputs > 1):
            Sorted target(s) classes.

    Note:
        binary classes only.
        classes should be extracted from train, not full dataset.

    """

    def __init__(self, threshold=None, **kwargs):
        # kwargs layer need to resolve hp together.
        classes = kwargs.get('classes', None)
        pos_labels = kwargs.get('pos_labels', None)
        pos_labels_ind = kwargs.get('pos_labels_ind', None)

        if not classes:
            raise ValueError("classes should be non-empty sequence.")
        classes = classes if isinstance(classes, list) else [classes]
        if any(not isinstance(i, np.ndarray) for i in classes):
            raise ValueError("'classes' should be array of shape (n_classes),"
                             " or a list of n_outputs such arrays if n_outputs > 1.")

        if any(len(i) > 2 for i in classes):
            raise ValueError('Currently only binary classification supported.')

        self.classes_ = classes
        self.pos_labels = pos_labels
        self.pos_labels_ind = pos_labels_ind
        self.threshold = threshold

    def fit(self, x, y, **fit_params):
        return self

    def predict(self, x):
        """
        Note:
            x = predict_proba
            in RF built-in:
                return self.classes.take(np.argmax(x, axis=1), axis=0)

        """

        # x = predict_proba
        if not isinstance(x, list):
            # Compliance to multi-output.
            x = [x]
            threshold = [self.threshold]
        else:
            threshold = self.threshold

        assert len(x) == len(self.classes_), "Multi-output inconsistent."

        res = []
        for i in range(len(x)):
            # Check that each train fold contain all classes, otherwise we
            # can`t predict_proba. We can`t reconstruct probabilities
            # (add zero), because don`t no which one class is absent.
            n_classes = x[i].shape[1]
            if n_classes != self.classes_[i].shape[0]:
                raise ValueError('Not all class labels  in train folds:\n'
                                 '    ThresholdClassifier can`t identify class probabilities.')

            if threshold[i] is None:
                # just take the max
                res.append(self.classes_[i].take(np.argmax(x[i], axis=1), axis=0))
            else:
                if n_classes > 2:
                    # Multi-class classification.
                    # Remain label with max prob.
                    mask = np.arange(n_classes) != self.pos_labels_ind
                    remain_x = x[i][..., mask]
                    remain_classes = self.classes_[i][mask]
                    neg_labels = remain_classes.take(np.argmax(remain_x, axis=1), axis=0)
                else:
                    # Binary classification.
                    mask = np.arange(n_classes) != self.pos_labels_ind
                    neg_labels = self.classes_[i][mask]
                res.append(np.where(x[i][..., self.pos_labels_ind] > threshold[i], [self.pos_labels], neg_labels))
        return res if len(res) > 1 else res[0]

    def predict_proba(self, x):
        return x


class SkippableTransformer(sklearn.decomposition.TruncatedSVD):
    def __init__(self, skip=False, n_components=2, algorithm="randomized", n_iter=5,
                 random_state=None, tol=0.):
        self.skip = skip
        super().__init__(n_components, algorithm, n_iter, random_state, tol)

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


class SMWrapper(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    """ A universal sklearn-style wrapper for statsmodels regressors """

    def __init__(self, model_class, fit_intercept=True):
        self.model_class = model_class
        self.fit_intercept = fit_intercept
        self.model_ = None
        self.results_ = None

    def fit(self, x, y):
        if self.fit_intercept:
            x = sm.add_constant(x)
        self.model_ = self.model_class(y, x)
        self.results_ = self.model_.fit()

    def predict(self, x):
        if self.fit_intercept:
            x = sm.add_constant(x)
        return self.results_.predict(x)


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
    except ValueError as e:
        y_pred_oof, index_oof = partial_cross_val_predict(*args, **kwargs)
    if _debug:
        assert np.array_equal(temp_pp, y_pred_oof)
        assert np.array_equal(temp_ind, index_oof)
    return y_pred_oof, index_oof


def partial_cross_val_predict(estimator, x, y, cv, fit_params=None,
                              method='predict_proba', **kwargs):
    """Extension to cross_val_predict for TimeSplitter."""
    if not fit_params:
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


# TODO: Better move to utills y_pred_to_probe, also get pos_labels_in
def prob_to_pred(y_pred_proba, th_, pos_labels, neg_label, pos_labels_ind):
    """Fix threshold on predict_proba"""
    y_pred = np.where(y_pred_proba[:, pos_labels_ind] > th_, [pos_labels], [neg_label])
    return y_pred


if __name__ == '__main__':
    pass
