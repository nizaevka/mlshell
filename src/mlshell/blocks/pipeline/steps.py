"""The :mod:`mlshell.pipeline.steps` contains unified pipeline steps."""


import inspect

import mlshell
import numpy as np
import pandas as pd
import sklearn
import sklearn.impute
import sklearn.compose

__all__ = ['Steps']


class Steps(object):
    """Unified pipeline steps.

    Parameters
    ----------
    estimator : :mod:`sklearn` estimator
        Estimator to use in the last step.
        If ``estimator_type=regressor``:
        ``sklearn.compose.TransformedTargetRegressor(regressor=`estimator`)``
        If ``estimator_type=classifier`` and ``th_step=True``:
        ``sklearn.pipeline.Pipeline(steps=[
            ('predict_proba',
                mlshell.model_selection.PredictionTransformer(`estimator`)),
            ('apply_threshold',
                mlshell.model_selection.ThresholdClassifier(threshold=0.5,
                                                   kwargs='auto')),
                    ])``
        If ``estimator_type=classifier`` and ``th_step=False``:
        ``sklearn.pipeline.Pipeline(steps=[('classifier', `estimator`)])``
    estimator_type : str {'classifier`, 'regressor'}, optional (default=None)
         Either regression or classification task. If None, get from
         :func:`sklearn.base.is_classifier` on ``estimator``.
    th_step : bool
        If True and ``estimator_type=classifier``: ``mlshell.model_selection.
        ThresholdClassifier`` sub-step added, otherwise ignored.

    Notes
    -----
    Assembling steps in class are made for convenience. Use steps property to
    access after initialization.  Only OneHot encoder and imputer steps are
    initially activated.
    By default, 4 parameters await for resolution ('auto'):

        'process_parallel__pipeline_categoric__select_columns__kw_args'
        'process_parallel__pipeline_numeric__select_columns__kw_args'
        'estimate__apply_threshold__threshold'
        'estimate__apply_threshold__params'

    Set corresponding parameters with ``set_params()`` to overwrite default in
    created pipeline or use :class:`mlshell.model_selection.Resolver` .

    'pass_custom' step allows brute force arbitrary parameters in uniform style
    with pipeline hp (as if score contains additional nested loops). Step name
    is hard-coded and could not be changed.

    'apply_threshold' allows grid search classification thresholds as pipeline
    hyper-parameter.

    'estimate' step should be the last.

    """
    _required_parameters = ['estimator', 'estimator_type']

    def __init__(self, estimator, estimator_type=None, th_step=False):
        if estimator_type is None:
            estimator_type = 'classifier' if sklearn.base.is_classifier(estimator)\
                else 'regressor'

        self._steps = [
            ('pass_custom',      mlshell.preprocessing.FunctionTransformer(func=self.scorer_kwargs, validate=False, skip=True, kw_args={})),
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
            ('reduce_dimensions', mlshell.decomposition.PCA(random_state=42, skip=True)),
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
                        params='auto', threshold=None)),
                    ])
        elif estimator_type == 'classifier' and not th_step:
            last_step = sklearn.pipeline.Pipeline(steps=[('classifier',
                                                          estimator)])
        else:
            raise ValueError(f"Unknown estimator type `{estimator_type}`.")

        if sklearn.base.is_classifier(estimator=last_step)\
                ^ (estimator_type == "classifier"):
            raise TypeError(f"{self.__class__.__name__}:"
                            f"{inspect.stack()[0][3]}:"
                            f" wrong estimator type: {last_step}")
        return last_step

    @property
    def steps(self):
        """list : access steps to pass in `sklearn.pipeline.Pipeline` ."""
        return self._steps

    def scorer_kwargs(self, x, **kw_args):
        """Mock function to custom kwargs setting.

        Parameters
        ----------
        x : :class:`numpy.ndarray` or :class:`pandas.DataFrame`
            Features of shape [n_samples, n_features].
        **kw_args : dict
            Step parameters. Could be extracted from pipeline in scorer if
            needed.

        Returns
        -------
        result: :class:`numpy.ndarray` or :class:`pandas.DataFrame`
            Unchanged ``x``.

        """
        return x

    def subcolumns(self, x, **kw_args):
        """Get sub-columns from x.

        Parameters
        ----------
        x : :class:`numpy.ndarray` or :class:`pandas.DataFrame`
            Features of shape [n_samples, n_features].
        **kw_args : dict
            Columns indices to extract: {'indices': array-like}.

        Returns
        -------
        result: :class:`numpy.ndarray` or :class:`pandas.DataFrame`
            Extracted sub-columns of ``x``.

        """
        indices = kw_args['indices']
        if isinstance(x, pd.DataFrame):
            return x.iloc[:, indices]
        else:
            return x[:, indices]

    def subrows(self, x):
        """Get rows from x."""
        # For example to delete outlier/anomalies.
        return x

    def bining_mask(self, x):
        """Get features indices which need bining."""
        # Use slice(0, None) to get all.
        return []


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


if __name__ == '__main__':
    pass
