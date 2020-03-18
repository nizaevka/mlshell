"""Module contains default configuration for user params and class to create default pipeline steps."""


from mlshell.libs import *
import mlshell.custom


DEFAULT_PARAMS = {
    'estimator_type': 'regressor',
    'main_estimator': sklearn.linear_model.LinearRegression(),
    'cv_splitter': sklearn.model_selection.KFold(shuffle=False),
    'metrics': {
        'score': sklearn.metrics.r2_score,
    },
    'split_train_size': 0.7,
    'hp_grid': {},
    'gs_flag': False,
    'estimator_fit_params': {},
    'del_duplicates': False,
    'debug_pipeline': False,
    'isneed_cache': False,
    'cache_update': True,
    'gs_verbose': 1,
    'n_jobs': 1,
    'isneeddump': False,
    'runs': None,
    'plot_analysis': False,
    'th_strategy': 0,
    'pos_label': 1,
    'train_file': None,
    'test_file': None,
    'rows_limit': None,
    'random_skip': False,
}
"""(dict): if user skip declaration for any parameter the default one will be used.

    estimator_type ('regressor' or 'classifier', optional (default='regressor')):
        Sklearn estimator type.
    main_estimator (``sklearn.base.BaseEstimator``, optional (default=sklearn.linear_model.LinearRegression())):
        Last step in pipeline.
    cv_splitter (``sklearn.model_selection`` splitter, optional (default=sklearn.model_selection.KFold(shuffle=False)):
        Yield train and test folds.
    score_metrics (dict of ``sklearn.metrics``, optional (default={'score': sklearn.metrics.r2_score})):
        Dict of metrics to be measured. Should consist 'score' key, which val is used for hp tunning.
    split_train_size (train_size for sklearn.model_selection.train_test_split, default=0.7):
        Split data on train and validation. It is possible to set 1.0 and CV on whole data.
    estimator_fit_params (dict, optional (default={})):
        Parametes will be passed to estimator.fit(**estimator_fit_params) method.
    del_duplicates (bool, optional (default=False)):
        If True remove duplicates rows from input data.
    debug_pipeline (bool, optional (default=False):
        If True fit pipeline and log exhaustive information.             
    isneed_cache (bool, optional (default=False):
        if True use ``memory`` argument in ``sklearn.pipeline.Pipeline`` to create steps` cache in ``temp/``.             
    cache_update (bool, optional (default=True):
        if True clean ``temp/`` before run.
    gs_flag (bool, optional (default=False)):
        if True tune hp in optimizer else just fit default pipeline.
    hp_grid (dict of params for sklearn hyper-parameter optimizers, optional (default={})):
        Full list see in ``mlshell.Steps`` class.
    gs_verbose (int (default=1)):
        `verbose` argument in optimizer if exist.
    n_job (int (default=1)):,
        `n_jobs` argument in optimizer if exist.
    isneeddump (bool, optional (default=False)):
        if True dump current estimator on disc. 
        Fitted if after fit method, with best params if gs_flag=True and hp_grid is not empty.
    runs (bool or None, optional (default=None)):
        Number of runs in optimizer.
        If None will be used hp_grid shapes multiplication.
    plot_analysis (bool, optional (default=False)):
        Use ``mlshell.GUI`` class for result visualisation.
    th_strategy ( 0,1,2,3, optional (default=0)):
        For classification only. For details see `Features <./Concepts.html#classification-threshold>`__.
    pos_label (int or str, optional (default=1)):
        For classification only. Label for positive class.
    
    train_file (bool, optional (default=None)):
        Relative path to csv file with train data (with targets) to cross-validate with reserved validation subset.
    test_file (bool, optional (default=None)):
        Relative path to csv file with new data (without targets) to predict.
    rows_limit (int or None, optional (default=None)):
        Number of lines get from input file, passes in GetData class (if argument is implemented).
    random_skip' (bool, optional (default=False)):
        If True and rows_limit=True get rows random from input, passed in GetData class (if argument is implemented).
 
"""

@nb.njit
def _isbinary_columns(arr: np.ndarray) -> np.ndarray:
    """Check if columns of 2d array is binary.

    Args:
        arr (2d np.ndarray[float64]): array to check
    Returns:
        result (1d np.ndarray[bool]): boolean array
            "i" element is True if column "i" is binary else False

    """
    is_binary = np.ones(arr.shape[1], dtype=np.bool, order='contigious')
    for ind in np.arange(is_binary.shape[0]):
        for v in np.nditer(arr):
            if v.item() != 0 and v.item() != 1:
                is_binary[ind] = False
                break
    return is_binary


class CreateDefaultPipeline(object):
    """Class to create default pipeline steps."""

    def __init__(self, categoric_ind_name, numeric_ind_name, set_custom_param, p):
        """

        Args:
            categoric_ind_name (dict): {column_index: ('feature_categr__name',['B','A','C']),}
            numeric_ind_name (dict):  {column_index: ('feature__name',),}
            set_custom_param (function): Function to pass GS parameters in Workflow instance `self` attributes.
            p (dict): User parameters, see `default_params <./mlshell.html#mlshell.default.default_params>`__.

        Notes:
            Target transformer for regression should be the last or absent.

        """
        self.default_steps = [
            # pass custom params to self-object for tune in brutforce external loop in GS
            ('pass_custom',      sklearn.preprocessing.FunctionTransformer(func=set_custom_param, validate=False)),
            ('select_rows',      sklearn.preprocessing.FunctionTransformer(func=self.subrows, validate=False)),   # delete outlier/anomalies
            ('process_parallel', sklearn.pipeline.FeatureUnion(transformer_list=[
                ('pipeline_categoric', sklearn.pipeline.Pipeline(steps=[
                   ('select_columns',      sklearn.preprocessing.FunctionTransformer(self.subcolumns, validate=False, kw_args={'indices': categoric_ind_name})),
                   ('encode_onehot',       mlshell.custom.preprocessing_OneHotEncoder(handle_unknown='ignore', categories=[list(range(len(i[1]))) for i in categoric_ind_name.values()], sparse=False, drop=None)),  # TODO: better specify categories from whole data? or 'auto', try drop
                ])),
                ('pipeline_numeric',   sklearn.pipeline.Pipeline(steps=[
                    ('select_columns',     sklearn.preprocessing.FunctionTransformer(self.subcolumns, validate=False, kw_args={'indices': numeric_ind_name})),
                    ('impute',             sklearn.pipeline.FeatureUnion([
                        ('indicators',         sklearn.impute.MissingIndicator(missing_values=np.nan)),
                        ('gaps',               sklearn.impute.SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0, copy=True)),  # there are add_indicator (False by default) option in fresh release
                        ])),
                    ('transform_normal',   mlshell.custom.preprocessing_SkippablePowerTransformer(method='yeo-johnson', standardize=False, copy=False, skip=True)),
                    ('scale_row_wise',     sklearn.preprocessing.FunctionTransformer(func=None, validate=False)),
                    ('scale_column_wise',  sklearn.preprocessing.RobustScaler(quantile_range=(0, 100), copy=False)),
                    ('add_polynomial',     sklearn.preprocessing.PolynomialFeatures(degree=1, include_bias=False)),  # x => degree=1 => x, x => degree=0 => []
                    ('compose_columns',    sklearn.compose.ColumnTransformer([
                        ("discretize",     sklearn.preprocessing.KBinsDiscretizer(n_bins=5, encode='onehot-dense', strategy='quantile'), self.bining_mask)], sparse_threshold=0, remainder='passthrough'))
                ])),
            ])),
            ('select_columns',   sklearn.feature_selection.SelectFromModel(estimator=mlshell.custom.CustomSelectorEstimator(estimator_type=p['estimator_type'], verbose=False, skip=True), prefit=False)),
            ('reduce_dimension', mlshell.custom.decomposition_CustomReducer(skip=True)),
            ('estimate', sklearn.compose.TransformedTargetRegressor(regressor=None, transformer=None, check_inverse=True)),
        ]

    def subcolumns(self, x, **kwargs):
        """Get subcolumns from x.
        
        Args: 
            x (np.ndarray or dataframe of shape=[[row],]): Input x.
            **kwargs: Should contain 'indices' key.

        Returns:
            result (np.ndarray or xframe): Subcolumns of x.
        """
        feat_ind_name = kwargs['indices']
        indices = list(feat_ind_name.keys())
        names = [i[0] for i in feat_ind_name.values()]
        if isinstance(x, pd.DataFrame):
            return x.loc[:, names]
        else:
            return x[:, indices]

    def subrows(self, x):
        """Get rows from x."""
        # [deprecated] delete very rare outliers in numeric features (10 sigm)
        # size_before = self.data.size
        # self.data = self.data[(np.abs(sc.stats.zscore(self.data[self.numeric_names])) < 10).all(axis=1)]
        # size_after = self.data.size
        # if size_after != size_before:
        #     self.logger.debug('MyWarning: delete outliers {}'.format(size_before - size_after))
        return x

    def numeric_mask(self, x):
        """Find numeric features` indices."""
        return np.invert(_isbinary_columns(x))

    def categor_mask(self, x):
        """Find binary features` indices."""
        return _isbinary_columns(x)

    def bining_mask(self, x):
        """Find features which need bining."""
        return []  # slice(0, None) for all


if __name__ == 'main':
    pass
