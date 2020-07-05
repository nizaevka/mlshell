"""The :mod:`mlshell.blocks.pipeline.steps` contains unified pipeline steps."""


import inspect

import mlshell
import numba
import numpy as np
import pandas as pd
import sklearn

__all__ = ['Steps']


@numba.njit
def _isbinary_columns(arr: np.ndarray) -> np.ndarray:
    """Check if columns in 2d array is binary.

    Parameters
    ----------
    arr: 2d array of shape [rows X columns]
        Array to check.

    Returns
    -------
    result: bool np.ndarray of shape [columns]
        "i" element is True if "i" column is binary else False.

    """
    is_binary = np.ones(arr.shape[1], dtype=np.bool, order='contigious')
    for ind in np.arange(is_binary.shape[0]):
        for v in np.nditer(arr):
            if v.item() != 0 and v.item() != 1:
                is_binary[ind] = False
                break
    return is_binary


class Steps(object):
    """Class to create pipeline steps.

    Parameters
    ----------
    estimator : sklearn estimator
        Estimator to use in the last step.
        If `estimator_type`='regressor':
        sklearn.compose.TransformedTargetRegressor(regressor=`estimator`)
        If `estimator_type`='classifier' and `th_step`=True:
        sklearn.pipeline.Pipeline(steps=[
            ('predict_proba',
                mlshell.model_selection.PredictionTransformer(`estimator`)),
            ('apply_threshold',
                mlshell.model_selection.ThresholdClassifier(threshold=0.5,
                                                   kwargs='auto')),
                    ])
        If `estimator_type`='classifier' and `th_step`=False:
        sklearn.pipeline.Pipeline(steps=[('classifier', `estimator`)])
    estimator_type : str
        'estimator` or 'regressor'.
    th_step : bool
        If True and 'classifier', `mlshell.model_selection.ThresholdClassifier`
        sub-step added.

    Notes
    -----
    Assemble steps in class are made for convenience.
    By default, 4 parameters await for resolution:
        'process_parallel__pipeline_categoric__select_columns__kwargs'
        'process_parallel__pipeline_numeric__select_columns__kwargs'
        'estimate__apply_threshold__threshold'
        'estimate__apply_threshold__kwargs'

    'pass_custom' step allows brute force arbitrary parameters in uniform style
    with pipeline hp, as if score contains additional nested loops (name is
    hard-coded).
    'apply_threshold' allows grid search classification thresholds as pipeline
    hyper-parameter.

    'estimate' step should be the last.

    """
    _required_parameters = ['estimator', 'estimator_type']

    def __init__(self, estimator, estimator_type, th_step=False):
        self._steps = [
            ('pass_custom',      mlshell.preprocessing.FunctionTransformer(func=self.set_scorer_kwargs, validate=False, skip=True)),
            ('select_rows',      mlshell.preprocessing.FunctionTransformer(func=self.subrows, validate=False, skip=True)),
            ('process_parallel', sklearn.pipeline.FeatureUnion(transformer_list=[
                ('pipeline_categoric', sklearn.pipeline.Pipeline(steps=[
                   ('select_columns',      mlshell.preprocessing.FunctionTransformer(self.subcolumns, validate=False, skip=False, kw_args='auto')),  # {'indices': dataset.meta['categoric_ind_name']}
                   ('encode_onehot',       mlshell.preprocessing.OneHotEncoder(handle_unknown='ignore', categories='auto', sparse=False, drop=None, skip=False)),  # x could be [].
                ])),
                ('pipeline_numeric',   sklearn.pipeline.Pipeline(steps=[
                    ('select_columns',     mlshell.preprocessing.FunctionTransformer(self.subcolumns, validate=False, skip=False, kw_args='auto')),  # {'indices': dataset.meta['numeric_ind_name']}
                    ('impute',             sklearn.pipeline.FeatureUnion([
                        ('indicators',         sklearn.impute.MissingIndicator(missing_values=np.nan, error_on_new=False)),
                        ('gaps',               sklearn.impute.SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0, copy=True)),
                        ])),
                    ('transform_normal',   mlshell.preprocessing.PowerTransformer(method='yeo-johnson', standardize=False, copy=False, skip=True)),
                    ('scale_row_wise',     mlshell.preprocessing.FunctionTransformer(func=None, validate=False, skip=True)),
                    ('scale_column_wise',  sklearn.preprocessing.RobustScaler(quantile_range=(0, 100), copy=False)),
                    ('add_polynomial',     sklearn.preprocessing.PolynomialFeatures(degree=1, include_bias=False)),  # x => degree=1 => x, x => degree=0 => []
                    ('compose_columns',    sklearn.compose.ColumnTransformer([
                        ("discretize",         sklearn.preprocessing.KBinsDiscretizer(n_bins=5, encode='onehot-dense', strategy='quantile'), self.bining_mask)], sparse_threshold=0, remainder='passthrough'))
                ])),
            ])),
            ('select_columns',   sklearn.feature_selection.SelectFromModel(estimator=CustomSelector(estimator_type=estimator_type, verbose=False, skip=True), prefit=False)),
            ('reduce_dimension', CustomReducer(skip=True)),
            ('estimate', self.last_step(estimator, estimator_type, th_step=th_step)),
        ]

    def last_step(self, estimator, estimator_type, th_step):
        """Prepare estimator step."""
        if estimator_type == 'regressor':
            last_step =\
                sklearn.compose.TransformedTargetRegressor(regressor=estimator)
        elif estimator_type == 'classifier' and th_step:
            last_step = sklearn.pipeline.Pipeline(steps=[
                ('predict_proba',
                    mlshell.model_selection.PredictionTransformer(
                        estimator)),
                ('apply_threshold',
                    mlshell.model_selection.ThresholdClassifier(
                        threshold=None, kwargs='auto')),
                    ])
        elif estimator_type == 'classifier' and not th_step:
            last_step = sklearn.pipeline.Pipeline(steps=[('classifier',
                                                          estimator)])
        else:
            raise ValueError(f"Unknown estimator type `{estimator_type}`.")

        if (sklearn.base.is_classifier(estimator=last_step)
                ^ estimator_type == "classifier"):
            raise TypeError(f"{self.__class__.__name__}:"
                            f"{inspect.stack()[0][3]}:"
                            f" wrong estimator type")
        return last_step

    @property
    def steps(self):
        """list : access steps to pass in `sklearn.pipeline.Pipeline`."""
        return self._steps

    def set_scorer_kwargs(self, x, **kwargs):
        """Mock function to allow custom kwargs setting.

        Parameters
        ----------
        x : numpy.ndarray or pandas.DataFrame of shape = rows x columns
            Features.
        **kwargs : dict
            Parameters to substitute in pipeline by hp identifier.

        Returns
        -------
        result: numpy.ndarray or pandas.DataFrame
            Unchanged `x`.

        """
        return x

    def subcolumns(self, x, **kwargs):
        """Get sub-columns from x.

        Parameters
        ----------
        x : numpy.ndarray or pandas.DataFrame of shape = rows x columns
            Features.
        **kwargs : dict {'indices': array-like}
            Parameters to substitute in pipeline by hp identifier.

        Returns
        -------
        result: numpy.ndarray or pandas.DataFrame
            Sub-columns of `x`.

        """
        indices = kwargs['indices']
        # [deprecated]
        # feat_ind_name = kwargs['indices']
        # indices = list(feat_ind_name.keys())
        # names = [i[0] for i in feat_ind_name.values()]
        if isinstance(x, pd.DataFrame):
            return x.iloc[:, indices]  # x.loc[:, names]
        else:
            return x[:, indices]

    def subrows(self, x):
        """Get rows from x."""
        # For example to delete outlier/anomalies.
        return x

    def bining_mask(self, x):
        """Find features which need bining."""
        # Use slice(0, None) if need all.
        return []

    def numeric_mask(self, x):
        """Find numeric features` indices.

        Inefficient (called every fit) local resolution for:
            'process_parallel__pipeline_numeric__select_columns__kwargs'
        """
        return np.invert(_isbinary_columns(x))

    def categor_mask(self, x):
        """Find binary features` indices.

        Inefficient (called every fit) local resolution for:
            'process_parallel__pipeline_categoric__select_columns__kwargs'
        """
        return _isbinary_columns(x)


class CustomSelector(sklearn.base.BaseEstimator):
    """Custom feature selector template."""

    def __init__(self, estimator_type='classifier', verbose=True,
                 skip=False):
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
        # sklearn.random_projection.johnson_lindenstrauss_min_dim
        # cluster.FeatureAgglomeration
        # sklearn.decomposition.PCA()
        return self

    def transform(self, x):
        if self.skip:
            return x
        x_transformed = None
        return x_transformed


if __name__ == '__main__':
    pass
