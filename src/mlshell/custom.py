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


class ThresholdClassifier(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    """Classify samples based on whether they are above of below `threshold`

    Note:
        binary classes only.

    """
    def __init__(self, classes, pos_label_ind='auto', pos_label='auto', neg_label='auto', threshold=0.5):
        if len(classes) > 2:
            raise ValueError('Currently only binary classification supported.')
        if not classes:
            raise ValueError('Classes should be non-empty sequence.')

        self.classes_ = classes
        self.pos_label_ind = len(classes)-1 if pos_label_ind == 'auto' else pos_label_ind
        self.pos_label = classes[-1] if pos_label == 'auto' else pos_label
        self.neg_label = classes[0] if neg_label == 'auto' else neg_label
        self.threshold = threshold

    def fit(self, x, y, **fit_params):
        return self

    def predict(self, x):
        # the implementation used here breaks ties differently
        # from the one used in RFs:
        # return self.classes.take(np.argmax(x, axis=1), axis=0)
        if x.shape[1] != self.classes_.shape[0]:
            raise MyException('MyError: not all class labels in train folds')
        return np.where(x[:, self.pos_label_ind] > self.threshold, [self.pos_label], [self.neg_label])

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


if __name__ == '__main__':
    pass
