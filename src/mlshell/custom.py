"""The module contains custom classes on base of sklearn classes

Note:
    All estimators/transformers should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``). Otherwise will be problem in GS.

"""


from abc import ABC


from mlshell.libs import *


class preprocessing_SkippablePowerTransformer(sklearn.preprocessing.PowerTransformer):
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


class preprocessing_OneHotEncoder(sklearn.preprocessing.OneHotEncoder):
    """Skippable version of OneHotEncoder."""

    def __init__(self, categories=None, drop=None, sparse=True, dtype=np.float64,
                 handle_unknown='error', skip=False):
        # can work if no categor functions x=[]
        self.skip = skip
        super().__init__(categories=categories, drop=drop, sparse=sparse, dtype=dtype,
                         handle_unknown=handle_unknown)
        # [deprecated] forbided example wwith kwargs
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


class decomposition_CustomReducer(sklearn.base.BaseEstimator):
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


class impute_CustomImputer(object):
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


class model_selection_CustomCV(sklearn.model_selection.BaseCrossValidator, ABC):
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
        # TODO:
        #   score all features between [0, 1]
        #   0 - delete, 1-remain, other-on threshold
        #   !! can be multiple target
        #   prepare
        #   * collinearity
        #   * log target
        #   * homo
        #   * correct HC1
        #   ols
        #   check Fisher criteria => delete non-informative features,
        #   univariate
        #   sparse model
        #   end-estimator will have regularization => sparse not necessary

        # self.univariate_test(x, y)
        # self.recursive_elimination(x, y)
        # self.sparse_selector(x, y)
        # self.feature_importances_ = np.full(x.shape[1], fill_value=1)
        return self

    # def univariate_test(self, x, y):
    #     # TODO: need non-collin, homo
    #     # statsmodels
    #     # sklearn analog
    #     if self.estimator_type == 'regressor':
    #         score_funcs = ['f_regression', 'mutual_info_regression']
    #     elif self.estimator_type == 'classificator':
    #         score_funcs = ['chi2', 'f_classif', 'mutual_info_classif']
    #     else:
    #         raise MyException('MyError: unknown estimator type')
    #     for score_func in score_funcs:
    #         if 'mutual' in score_func:
    #             modes = ['percentile', 'k_best']
    #         else:
    #             modes = ['percentile', 'k_best', 'fpr', 'fdr', 'fwe']
    #         for mode in modes:
    #             # param default
    #             temp = sklearn.feature_selection.GenericUnivariateSelect(score_func=score_func, mode=mode).fit(x, y)
    #             self.logger.info('Univariate analysis: mode:{} score_func:{}:\n'
    #                              'scores:\n'
    #                              '{}\n'
    #                              'pvalues:\n'
    #                              'p-values\n'
    #                              '{}\n'
    #                              '{}'.format(mode, score_func, temp.scores_, temp.pvalues_, self.delimeter))

    # def recursive_elimination(self, x, y):
    #     # recursive CV elimination analysis
    #     if self.estimator_type == 'regressor':
    #         simple_estimator = sklearn.linear_model.LinearRegression()
    #     elif self.estimator_type == 'classificator':
    #         simple_estimator = sklearn.linear_model.LogisticRegression()  # sklearn.svm.SVC(kernel="linear")
    #     else:
    #         raise MyException('MyError: unknown estimator type')

    #     rfecv = feature_selection.RFECV(estimator=simple_estimator, step=1, cv=self.cv, scoring=self.score)
    #     rfecv.fit(x, y)
    #     self.logger.info("Recursive CV elimination: remain features={}"
    #                      " ranking=\n"
    #                      "{}"
    #                      "mask=\n"
    #                      "{}"
    #                      "grid_scores_=\n"
    #                      "{}".format(rfecv.n_features_, rfecv.ranking_, rfecv.suppor_, rfecv.grid_scores_))

    # def sparse_selector(self, x, y):
    #     # SelectFromModel

    #     if self.estimator_type == 'regressor':
    #         sparse_estimator = linear_model.LassoCV(cv=self.cv, random_state=42)
    #     elif self.estimator_type == 'classificator':
    #         sparse_estimator = linear_model.LogisticRegressionCV(cv=self.cv, random_state=42)
    #     else:
    #         raise MyException('MyError: unknown estimator type')

    #     sparse_estimator = sparse_estimator.fit(x, y)
    #     selector_sparse = feature_selection.SelectFromModel(sparse_estimator, prefit=True)


class PredictionTransformer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin, sklearn.base.MetaEstimatorMixin):
    """Class to add brutforce of th in GridSearch"""

    def __init__(self, clf):
        """Replaces all features with `clf.predict_proba(x)`"""
        self.clf = clf

    def fit(self, x, y, **fit_params):
        self.clf.fit(x, y, **fit_params)
        return self

    def transform(self, x):
        return self.clf.predict_proba(x)


class ThresholdClassifier(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    """Classify samples based on whether they are above of below `threshold`

    work only with binary
    pos_label=1

    """

    def __init__(self, classes_, pos_label_ind, pos_label, neg_label, threshold=0.5):
        self.classes_ = classes_
        self.pos_label_ind = pos_label_ind
        self.pos_label = pos_label
        self.threshold = threshold
        self.neg_label = neg_label

    def fit(self, x, y, **fit_params):
        return self

    def predict(self, x):
        # the implementation used here breaks ties differently
        # from the one used in RFs:
        # return self.classes_.take(np.argmax(x, axis=1), axis=0)
        if x.shape[1] != self.classes_.shape[0]:
            raise MyException('MyError: not all class labels in train folds')
        return np.where(x[:, self.pos_label_ind] > self.threshold, [self.pos_label], [self.neg_label])  # *self.classes_

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


if __name__ == 'main':

    # test threshold tunning, dynamic pass user params in score in GS
    def set_scorer_param(self, *arg, **kwarg):
        # self.logger.info(kwarg)
        for k, v in kwarg.items():
            setattr(self, k, v)
        return arg[0]  # pass x futher


    pipe = sklearn.model_selection.make_pipeline(
        sklearn.preprocessing.FunctionTransformer(set_scorer_param, validate=False),
        PredictionTransformer(sklearn.ensemble.RandomForestClassifier()),
        ThresholdClassifier(2, -1, 1, 1))

    pipe_param_grid = {'predictiontransformer__clf__max_depth': [1, 2, 5, 10, 20, 30, 40, 50],
                       'predictiontransformer__clf__max_features': [8, 16, 32, 64, 80, 100],
                       'thresholdclassifier__threshold': np.linspace(0., 1., num=100)}
