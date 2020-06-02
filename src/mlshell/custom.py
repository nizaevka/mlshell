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
    def __init__(self, threshold=None, classes=None, pos_labels=None, pos_labels_ind=None):
        """Classify samples based on whether they are above of below `threshold`.

        Args:
            classes (array of shape (n_classes), or a list of n_outputs such arrays if n_outputs > 1):
                Sorted target(s) classes.

        Note:
            binary classes only.
            classes should be extracted from train, not full dataset.

        """
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
        if not isinstance(x, list):
            # Compliance to multioutput.
            x = [x]
        assert len(x) == len(self.classes_), "Multi-output inconsistent."

        res = []
        for i in range(len(x)):
            # Check that each train fold contain all classes, otherwise we can`t predict_proba.
            # We can`t reconstruct probabilities(add zero), because don`t no which one class is absent.
            n_classes = x[i].shape[1]
            if n_classes != self.classes_[i].shape[0]:
                raise ValueError('Not all class labels  in train folds:\n'
                                 '    ThresholdClassifier can`t identify class probabilities.')

            if self.threshold is None:
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
                res.append(np.where(x[i][..., self.pos_labels_ind] > self.threshold, [self.pos_labels], neg_labels))
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
    """Function to make bind OOF prediction/predict_proba.
    Args:
        args
        kwargs
    Returns:
        folds_predict_proba (2d np.ndarray): OOF probability predictions [n_test_samples x n_classes].
        folds_test_index (1d np.ndarray): test indices for OOF subset (reseted, not raw).
        y_true (1d np.ndarray): test for OOF subset (for Kfold whole dataset).
    TODO:
        in some fold could be not all classes, need to check.
    """
    # dev check for custom OOF
    debug = kwargs.get('cross_val_predict_debug', False)
    temp_pp = None
    temp_ind = None
    try:
        folds_predict_proba = sklearn.model_selection.cross_val_predict(*args, **kwargs)
        folds_test_index = np.arange(0, folds_predict_proba.shape[0])
        if debug:
            temp_pp = folds_predict_proba
            temp_ind = folds_test_index
            raise ValueError('debug')
    except ValueError as e:
        folds_predict_proba, folds_test_index = _cross_val_predict_extension(*args, **kwargs)
    if debug:
        assert np.array_equal(temp_pp, folds_predict_proba)
        assert np.array_equal(temp_ind, folds_test_index)
    # y != y_true for TimeSplitter
    y = kwargs.get('y')
    y_true = y.values[folds_test_index] if hasattr(y, 'loc') else y[folds_test_index]
    return folds_predict_proba, folds_test_index, y_true


def _cross_val_predict_extension(estimator, x, y=None, cv=None, fit_params=None, method='predict_proba', **kwargs):
    """Extension of cross_val_predict for TimeSplitter.
    Note:
        TimeSplitter has no prediction at first fold.
    """
    if method is not 'predict_proba':
        raise ValueError("Currently only 'predict_proba' method supported.")

    # self.logger.warning('Warning: {}'.format(e))
    folds_predict_proba = []  # list(range(self.cv_n_splits))
    folds_test_index = []  # list(range(self.cv_n_splits))
    # th_ = [[2, 1. / self.n_classes] for i in self.classes_]  # init list for th_ for every class
    ind = 0
    for fold_train_index, fold_test_index in cv.split(x):
        # stackingestimator__sample_weight=train_weights[fold_train_subindex]
        if hasattr(x, 'loc'):
            estimator.fit(x.loc[x.index[fold_train_index]],
                          y.loc[y.index[fold_train_index]],
                          **fit_params)
            # in order of pipeline.classes_
            fold_predict_proba = estimator.predict_proba(x.loc[x.index[fold_test_index]])
        else:
            estimator.fit(x[fold_train_index], y[fold_train_index], **fit_params)
            # in order of pipeline.classes_
            fold_predict_proba = estimator.predict_proba(x[fold_test_index])
        # merge th_ for class
        # metrics.roc_curve(y[fold_test_index], y_test_prob, pos_labels=self.pos_labels)
        # th_[self.pos_labels].extend(fold_th_)
        folds_test_index.extend(fold_test_index)
        folds_predict_proba.extend(fold_predict_proba)
        ind += 1
    folds_predict_proba = np.array(folds_predict_proba)
    folds_test_index = np.array(folds_test_index)
    # delete duplicates
    # for i in range(self.n_classes):
    #    th_[i] = sorted(list(set(th_[i])), reverse=True)
    return folds_predict_proba, folds_test_index

if __name__ == '__main__':
    pass
