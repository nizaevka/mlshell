"""Module contains default configuration for user params and class to create default pipeline steps."""


from mlshell.libs import *
import mlshell.custom


DEFAULT_PARAMS = {
    # data operation automative, create, load pipelines too
    # metrics better with list metrics_ids instead of metrics_id
    # possibility of multiple pipelines, metrics lists, gs_conf in one workflow!
    'workflow':
        {
            'endpoint_id': 'auto',
            'steps': {
                'fit': 1,
                'dump': 0,
                'validate': 1,
                'gui': 1,
                'predict': 1,
            },
        },
    'endpoint': {
        'default': {
            'pipeline_id':'auto',
            # 'endpoint_id':'auto',
            'get': {'class': 'default'},
            'preprocess': {'class': 'default'},
            'split': {'func':'default'},

            'load': {'func': 'default'},
            'dump': {'func': 'default'},
            'create': {'func': 'default'},
            'fit': {'func':'default',
                    'data_id': 'train',
                    'gs_flag': False,
                    'gs_id': 'auto',
                    'debug':False},
            'validate': {'data_id': 'train', 'metrics_id': 'auto'},
            'gui': {'class': 'default', 'base_sort': False, 'data_id': 'train'},
            'predict': {'function': None, 'data_id': 'test'},
        },
    },
    'pipeline': {
        'default': {
            'estimator': sklearn.linear_model.LinearRegression(),
            'type': 'regressor',
            'steps_cache': None,
            'steps': 'default',
            'fit_params': {},
            'filepath': None
        },
    },
    'metric': {
        'default': {
            'score': (sklearn.metrics.r2_score, {'greater_is_better': True}),
        },
    },
    'gs': {
        'default' :{
            # 'flag': True,
            'hp_grid': {},
            'n_iter': None,
            'scoring': ['score'],
            'n_jobs': 1,
            'refit': 'score',
            'cv': sklearn.model_selection.KFold(n_splits=3, shuffle=True),
            'verbose': 1,
            'pre_dispatch': 'n_jobs',
            # not necessary to specify all
            #'random_state': None,
            #'error_score': np.nan,
            #'return_train_score': True,
        },
    },
    'data': {
        'default': {
            'get': {},
            'preprocess': {},
            'unify': True,
            'cache': False,
            'split': False,
        },
    },
    'seed': 42,
}

# [deprecated]
# DEFAULT_PARAMS = {
#     'pipeline__estimator': sklearn.linear_model.LinearRegression(),
#     'pipeline__type': 'regressor',
#     'pipeline__fit_params': {},
#     'pipeline__steps': None,
#     'pipeline__debug': False,
#     'pipeline_cache': None,
#
#     'metrics': {
#         'score': (sklearn.metrics.r2_score, {'greater_is_better': True}),
#     },
#     'gs__flag': False,
#     'gs__splitter': sklearn.model_selection.KFold(shuffle=False),
#     'gs__hp_grid': {},
#     'gs__verbose': 1,
#     'gs__n_jobs': 1,
#     'gs__pre_dispatch': 'n_jobs',
#     'gs__refit': 'score',
#     'gs__runs': None,
#     'gs__metrics': ['score'],
#
#     'data__split_train_size': 0.7,
#     'data__del_duplicates': False,
#     'data__train__args': [],
#     'data__train__kw_args': {},
#     'data__test__args': [],
#     'data__test__kw_args': {},
#
#     'th__pos_label': 1,
#     'th__strategy': 0,
#     'th__samples': 10,
#     'th__plot_flag': False,
#
#     'cache__pipeline': False,
#     'cache__unifier': False,
#
#     'seed': 42,
# }
"""(dict): if user skip declaration for any parameter the default one will be used.

    pipeline__estimator (``sklearn.base.BaseEstimator``, optional (default=sklearn.linear_model.LinearRegression())):
        Last step in pipeline.
    pipeline__type ('regressor' or 'classifier', optional (default='regressor')):
        Last step estimator type.
    pipeline__fit_params (dict, optional (default={})):
        | Parametes will be passed to estimator.fit( ** estimator_fit_params) method.
        | For example: {'estimate__classifier__early_stopping_rounds': 200, 'estimate__classifier__eval_metric': 'auc'}
    pipeline__steps (custom class to create pipeline steps, optional (default=None))
        Will replace ``mlshell.default.CreateDefaultPipeline`` if set, should have .get_steps() method.
    pipeline__debug (bool, optional (default=False):
        If True fit pipeline on <=1k subdata and log exhaustive information.     
    metrics (dict of ``sklearn.metrics``, optional (default={'score': sklearn.metrics.r2_score})):
        Dict of metrics to be measured. Should consist 'score' key, which val is used for sort hp tuning results.
    gs__flag (bool, optional (default=False)):
        if True tune hp in optimizer and fit best just else fit pipeline with zero-position hp_grid.
    gs__splitter (``sklearn.model_selection`` splitter, optional (default=sklearn.model_selection.KFold(shuffle=False)):
        Yield train and test folds.
    gs__hp_grid (dict of params for sklearn hyper-parameter optimizers, optional (default={})):
        Full list see in ``mlshell.default.CreateDefaultPipeline`` class.
    gs__verbose (int (default=1)):
        `verbose` argument in optimizer.
    gs__n_job (int (default=1)):,
        ``n_jobs`` argument in optimizer.
    gs__pre_dispatch  (None or int or string, optional (default='n_jobs'))
        `pre_dispatch` argument in optimizer.
    gs__runs (bool or None, optional (default=None)):
        Number of runs in optimizer, hould be set if any hp_grid key is probability distribution.
        If None will be used hp_grid shapes multiplication.
    gs_metrics (list, optional (default=['score']))
        Sublist of ``metrics`` to evaluate in grid search.
        Always should contain 'score'.
    data__split__train_size (train_size for sklearn.model_selection.train_test_split, default=0.7):
        Split data on train and validation. It is possible to set 1.0 and CV on whole data (validation=train).
    data__del_duplicates (bool, optional (default=False)):
        If True remove duplicates rows from input data before pass to pipeline (workflow class level).
    data__train__args/data__train__kw_args (list, (default=[])
        Specify args to pass in user-defined classes.GetData class constructor.
        Typically there are contain path to files, index_column name, rows read limit. 
        For example see `Examples <./Examples.html>`__.                     
    data__test__args/data__test__kw_args (dict, (default={})
        Specify kw_args to pass in user-defined classes.GetData class constructor.
        Typically there are contain index_column name, rows read limit. 
        For example see `Examples <./Examples.html>`__.                     
    th__pos_label (int or str, optional (default=1)):
        For classification only. Label for positive class.
    th__strategy ( 0,1,2,3, optional (default=0)):
        | For classification only. 
        | ``th_`` tuning strategy.
        | For details see `Concepts <./Concepts.html#classification-threshold>`__.   
    th__samples (int, optional (default=100)):
        | For classification only. 
        | Number of ``th_`` values to brutforce for roc_curve based th_strategy (1.2/2.2/3.2).
    th__plot_flag (bool, optional (default=False):
        For ``th_strategy`` (1.2/2.2/3.2) plot ROC curve and trp/(tpr+fpr) vs ``th_`` with ``th_`` search range marks.
    cache__pipeline (bool, optional (default=False):
        if True, use ``memory`` argument in ``sklearn.pipeline.Pipeline``, cache steps` in ``result/cache/pipeline``.             
        If 'update', update cache files.
        If false, not use cache.
    cache__unifier (bool, optional (default=False):
        If True, cache input after workflow.unify_data ``result/cache/unifier/``, use that cache next time if available.
        If 'update', update cache file.
        If false, not use cache.
    seed (None or int, optional(default=42)):
        workflow random state for random.seed(42), numpy.random.seed(42).

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

    def __init__(self, categoric_ind_name, numeric_ind_name, params):
        """
        Args:
            categoric_ind_name (dict): {column_index: ('feature_categor__name', ['B','A','C']),}
            numeric_ind_name (dict):  {column_index: ('feature__name',),}
            params (dict): User parameters, see `default_params <./mlshell.html#mlshell.default.default_params>`__.

        Notes:
            Target transformer for regression should be the last or absent.
            Pass custom should be the first or absent.
            Estimator step is auto-filled in workflow.

        """
        self._steps = [
            ('pass_custom',      sklearn.preprocessing.FunctionTransformer(func=self.set_scorer_kw_args, validate=False)),
            ('select_rows',      sklearn.preprocessing.FunctionTransformer(func=self.subrows, validate=False)),
            ('process_parallel', sklearn.pipeline.FeatureUnion(transformer_list=[
                ('pipeline_categoric', sklearn.pipeline.Pipeline(steps=[
                   ('select_columns',      sklearn.preprocessing.FunctionTransformer(self.subcolumns, validate=False, kw_args={'indices': categoric_ind_name})),
                   ('encode_onehot',       mlshell.custom.SkippableOneHotEncoder(handle_unknown='ignore', categories='auto', sparse=False, drop=None, skip=False)),
                ])),
                ('pipeline_numeric',   sklearn.pipeline.Pipeline(steps=[
                    ('select_columns',     sklearn.preprocessing.FunctionTransformer(self.subcolumns, validate=False, kw_args={'indices': numeric_ind_name})),
                    ('impute',             sklearn.pipeline.FeatureUnion([
                        ('indicators',         sklearn.impute.MissingIndicator(missing_values=np.nan, error_on_new=False)),
                        ('gaps',               sklearn.impute.SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0, copy=True)),
                        ])),
                    ('transform_normal',   mlshell.custom.SkippablePowerTransformer(method='yeo-johnson', standardize=False, copy=False, skip=True)),
                    ('scale_row_wise',     sklearn.preprocessing.FunctionTransformer(func=None, validate=False)),
                    ('scale_column_wise',  sklearn.preprocessing.RobustScaler(quantile_range=(0, 100), copy=False)),
                    ('add_polynomial',     sklearn.preprocessing.PolynomialFeatures(degree=1, include_bias=False)),  # x => degree=1 => x, x => degree=0 => []
                    ('compose_columns',    sklearn.compose.ColumnTransformer([
                        ("discretize",     sklearn.preprocessing.KBinsDiscretizer(n_bins=5, encode='onehot-dense', strategy='quantile'), self.bining_mask)], sparse_threshold=0, remainder='passthrough'))
                ])),
            ])),
            ('select_columns',   sklearn.feature_selection.SelectFromModel(estimator=mlshell.custom.CustomSelectorEstimator(estimator_type=params['pipeline__type'], verbose=False, skip=True), prefit=False)),
            ('reduce_dimension', mlshell.custom.CustomReducer(skip=True)),
            ('estimate', sklearn.compose.TransformedTargetRegressor(regressor=None, transformer=None, check_inverse=True)),
        ]

    def get_steps(self):
        """Pipeline steps getter."""
        return self._steps

    def set_scorer_kw_args(self, *args, **kw_args):
        """Mock function to set custom kw_args."""
        # pass x futher
        return args[0]

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
        # delete outlier/anomalies
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


if __name__ == '__main__':
    pass
