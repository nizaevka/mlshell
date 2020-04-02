"""Module contains default configuration for user params and class to create default pipeline steps.

TODO:
    better specify categories from whole data (don`t available when use_unifier_cache) or 'auto'
    [list(range(len(i[1]))) for i in categoric_ind_name.values()]

"""


from mlshell.libs import *
import mlshell.custom


DEFAULT_PARAMS = {
    'pipeline': {
        'estimator': sklearn.linear_model.LinearRegression(),
        'type': 'regressor',
        'fit_params': {},
        'steps': None,
        'debug': False,
    },
    'metrics': {
        'score': (sklearn.metrics.r2_score, {'greater_is_better': True}),
    },
    'gs': {
        'flag': False,
        'splitter': sklearn.model_selection.KFold(shuffle=False),
        'hp_grid': {},
        'verbose': 1,
        'n_jobs': 1,
        'runs': None,
        'metrics': ['score'],
    },
    'th': {
        'strategy': 0,
        'pos_label': 1,
        'samples': 10,
        'plot_flag': False,
    },
    'cache': {
        'pipeline': False,
        'unifier': False,
    },
    'data': {
        'train': {
            'args': [],
            'kw_args': {},
        },
        'test': {
            'args': [],
            'kw_args': {},
        },
        'split_train_size': 0.7,
        'del_duplicates': False,
    }
}
# 'main_estimator' => 'pipeline_estimator'
# 'estimator_type' => 'pipeline_type'
# 'estimator_fit_params' => 'pipeline_fit_params'
# 'pipeline' => 'pipeline_steps'
# 'cv_splitter' => 'gs_splitter'
# 'hp_grid' => 'gs_hp_grid'
# 'debug_pipeline' => 'pipeline_debug'
# 'n_jobs' => 'gs_n_jobs'
# 'runs' => 'gs_runs'
# 'th_points_number' => 'th_samples'
# 'pos_label' => 'th_pos_label'
# 'get_data' => 'data'
# 'model_dump' => 'deprecated'
# 'del_duplicates' => 'data_del_duplicates'
# 'update_pipeline_cache','update_pipeline_cache' => 'cache_pipeline' False,None -True,'update'
# 'use_unifier_cache', 'update_unifier_cache' => 'cache_unifier'  False,None -True,'update'
# 'split_train_size' => 'data_split_train_size'


DEFAULT_PARAMS = {
    'estimator_type': 'regressor',
    'main_estimator': sklearn.linear_model.LinearRegression(),
    'estimator_fit_params': {},
    'pipeline': None,
    'debug_pipeline': False,
    'use_pipeline_cache': False,
    'update_pipeline_cache': False,
    'model_dump': False,

    'cv_splitter': sklearn.model_selection.KFold(shuffle=False),
    'metrics': {
        'score': (sklearn.metrics.r2_score, {'greater_is_better': True}, True),
    },
    'split_train_size': 0.7,

    'hp_grid': {},
    'gs_flag': False,
    'gs_verbose': 1,
    'n_jobs': 1,
    'runs': None,

    'pos_label': 1,
    'th_strategy': 0,
    'th_points_number': 10,
    'th_plot_flag': False,


    'use_unifier_cache': False,
    'update_unifier_cache': False,
    'del_duplicates': False,
    'get_data': {
        'train': {
            'args': [],
            'kw_args': {},
        },
        'test': {
            'args': [],
            'kw_args': {},
        },
    },

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
        | Parametes will be passed to estimator.fit(**estimator_fit_params) method.
        | For example: {'estimate__classifier__early_stopping_rounds': 200, 'estimate__classifier__eval_metric': 'auc'}
    del_duplicates (bool, optional (default=False)):
        If True remove duplicates rows from input data.
    pipeline (custom class to create pipeline steps, optional (default=None))
        Will replace default_pipeline if necessary, should create ``self.steps`` in ``__init__``.
    debug_pipeline (bool, optional (default=False):
        If True fit pipeline and log exhaustive information.             
    use_unifier_cache (bool, optional (default=False):
        if True, cache input after workflow.unify_data ``result/cache/unifier/``, use that cache next time if available.             
    update_unifier_cache (bool, optional (default=True):
        if True update ``result/cache/unifier/<cache_file>`` in workflow.unify_data.
    use_pipeline_cache (bool, optional (default=False):
        if True, use ``memory`` argument in ``sklearn.pipeline.Pipeline``, cache steps` in ``result/cache/pipeline``.             
    update_pipeline_cache (bool, optional (default=True):
        if True update ``result/cache/pipeline``.
    gs_flag (bool, optional (default=False)):
        if True tune hp in optimizer else just fit default pipeline.
    hp_grid (dict of params for sklearn hyper-parameter optimizers, optional (default={})):
        Full list see in ``mlshell.Steps`` class.
    gs_verbose (int (default=1)):
        `verbose` argument in optimizer if exist.
    n_job (int (default=1)):,
        `n_jobs` argument in optimizer if exist.
    model_dump (bool, optional (default=False)):
        if True dump current estimator on disc. 
        Fitted if after fit method, with best params if gs_flag=True and hp_grid is not empty.
    runs (bool or None, optional (default=None)):
        Number of runs in optimizer.
        If None will be used hp_grid shapes multiplication.
    pos_label (int or str, optional (default=1)):
        For classification only. Label for positive class.
    th_strategy ( 0,1,2,3, optional (default=0)):
        | For classification only. 
        | Mlshell support multiple strategy for ``th_`` tuning.
        | For details see `Concepts <./Concepts.html#classification-threshold>`__.                    
    th_points_number (int, optional (default=100)):
        | For classification only. 
        | Number of th_ values to brutforce for roc_curve based th_strategy (1/2/3.2).
    th_plot_flag (bool, optional (default=False):
        For ``th_strategy`` (1/2/3.2) plot ROC curve and trp/(tpr+fpr) vs ``th_`` with ``th_`` search range marks.
    get_data (dict, (default={'train': {'args': [], 'kw_args': {}}, 'test': {'args': [], 'kw_args': {}}})):
        Specify args and kw_args to pass in user-defined classes.GetData class constructor.
        Usually there are contain path to files, index_column name, rows read limit. 
        For example see `Examples <./Examples.html>`__.                     
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
        self.steps = [
            # pass custom params to self-object for tune in brutforce external loop in GS
            ('pass_custom',      sklearn.preprocessing.FunctionTransformer(func=set_custom_param, validate=False)),
            ('select_rows',      sklearn.preprocessing.FunctionTransformer(func=self.subrows, validate=False)),   # delete outlier/anomalies
            ('process_parallel', sklearn.pipeline.FeatureUnion(transformer_list=[
                ('pipeline_categoric', sklearn.pipeline.Pipeline(steps=[
                   ('select_columns',      sklearn.preprocessing.FunctionTransformer(self.subcolumns, validate=False, kw_args={'indices': categoric_ind_name})),
                   ('encode_onehot',       mlshell.custom.preprocessing_OneHotEncoder(handle_unknown='ignore', categories='auto', sparse=False, drop=None)),
                ])),
                ('pipeline_numeric',   sklearn.pipeline.Pipeline(steps=[
                    ('select_columns',     sklearn.preprocessing.FunctionTransformer(self.subcolumns, validate=False, kw_args={'indices': numeric_ind_name})),
                    ('impute',             sklearn.pipeline.FeatureUnion([
                        ('indicators',         sklearn.impute.MissingIndicator(missing_values=np.nan, error_on_new=False)),
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
