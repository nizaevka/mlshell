"""The module contains default configuration and pipeline steps."""


from mlshell.libs import *
import mlshell.custom
import mlshell.optimize
import mlshell.validate


WORKFLOW = {
    'default': {
        'init': {},
        'producer': mlshell.Workflow,
        'global': {},
        'patch': {},
        'priority': 1,
        'steps': [
            ('fit', {
                'pipeline_id': None,
                'dataset_id': 'train',
                'fit_params': {},
                'hp': {},
                'resolve_params': {},
            },),
            ('optimize', {
                'optimizer': mlshell.optimize.RandomizedSearchOptimizer,  # optimizer
                'validator': mlshell.validate.Validator,
                'pipeline_id': None,  # multiple pipeline? no, user can defined separately if really needed
                'dataset_id': 'train',
                'gs_params': {
                   'hp_grid': {},
                   'n_iter': None,  # ! my resolving (1, hp_grid number), otherwise 'NoneType' object cannot be interpreted as an integer
                   'scoring': None,  # no resolving (default estimator scoring)
                   'n_jobs': 1,
                   'refit': None, # no resolving
                   'cv': sklearn.model_selection.KFold(n_splits=3, shuffle=True),
                   'verbose': 1,
                   'pre_dispatch': 'n_jobs',
                   # TODO: for thresholdoptimizer, also need add pass_custom step.
                   #   so here params to mock.
                   # 'th_name':
                },
                'fit_params': {},
                'resolve_params': {
                    'estimate__apply_threshold__threshold': {
                        'resolver': None,  # if None, use pipeline default resolver
                        'samples': 10,
                        'plot_flag': False,
                    },
                },
            },),
            ('dump', {'pipeline_id': None}),
            ('validate', {
                'dataset_id': 'train',
                'validator': None,
                'metric': None,
                'pos_label': None,  # if None, get -1
                'pipeline_id': None,
            },),
            ('plot', {
                'plotter': None,  # gui
                'pipeline_id': None,
                'hp_grid': {},
                'dataset_id': 'train',
                'base_sort': False,
                # TODO: [beta]
                # 'dynamic_metric': None,
            }),
            ('predict', {
                'dataset_id': 'test',
                'pipeline_id': None,
            }),
            # TODO: [beta] Free memory.
            #   ('init',),
            #   ('reset',),
        ],
    },
    # TODO: [beta] DL workflow.
    #   'pytorch': {},
}
""""""


PIPELINES = {
    'default': {
        'init': mlshell.Pipeline,
        'producer': mlshell.PipeProducer,
        'global': {},
        'patch': {},
        'priority': 0,
        'steps': [
            ('create', {
                'cache': None,
                'steps': None,
                'estimator': sklearn.linear_model.LinearRegression(),
                'estimator_type': 'regressor',
                # 'th_strategy': None,
            },),
            ('resolve', {},),
                # [deprecated]  should be setted 'auto'/['auto'], by default only for index
                #  only if not setted
                # 'hp': {
                #     'process_parallel__pipeline_categoric__select_columns__kwargs',
                #     'process_parallel__pipeline_numeric__select_columns__kwargs',
                #     'estimate__apply_threshold__threshold'}
                # },
        ],
    },
}


METRICS = {
    'classifier': {
        'init': None,
        'producer': mlshell.MetricProducer,
        'global': {},
        'patch': {},
        'priority': 0,
        'steps': [
            ('make_scorer', {
                'func': sklearn.metrics.accuracy_score,
                'kwargs': {'greater_is_better': True},
            }),
        ],
    },
    'regressor': {
        'init': None,
        'producer': mlshell.MetricProducer,
        'global': {},
        'patch': {},
        'priority': 0,
        'steps': [
            ('make_scorer', {
                'func': sklearn.metrics.r2_score,
                'kwargs': {'greater_is_better': True},
            }),
        ],
    }
}


DATASETS = {
    'default': {
        'init': mlshell.Dataset(),
        'class': mlshell.DataProducer,
        'global': {},
        'patch': {},
        'priority': 0,

        'steps': [
            ('load_cache', {'prefix': None},),
            ('get', {},),
            ('preprocess', {'categor_names': [], 'target_names': [], 'pos_labels': []},),
            ('info', {},),
            ('unify', {},),
            ('split', {},),
            ('dump_cache', {'prefix': None},),
        ],
    },
}


DEFAULT_PARAMS = {
    'pipeline': PIPELINES,
    'dataset': DATASETS,
    'metric': METRICS,
    'workflow': WORKFLOW,
}
"""Default sections for ML task.

For ML task, common sections would be:
* create/read pipelines and datasets objects.
* create workflow class and call methods with pipeline/dataset as argument.

"""


# TODO: Move out to related methods.
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
    data__train__args/data__train__kwargs (list, (default=[])
        Specify args to pass in user-defined classes.GetData class constructor.
        Typically there are contain path to files, index_column name, rows read limit. 
        For example see `Examples <./Examples.html>`__.                     
    data__test__args/data__test__kwargs (dict, (default={})
        Specify kwargs to pass in user-defined classes.GetData class constructor.
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

    def __init__(self, estimator=None, estimator_type=None, th_strategy=None, **kwargs):
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
            ('pass_custom',      sklearn.preprocessing.FunctionTransformer(func=self.set_scorer_kwargs, validate=False)),
            ('select_rows',      sklearn.preprocessing.FunctionTransformer(func=self.subrows, validate=False)),
            ('process_parallel', sklearn.pipeline.FeatureUnion(transformer_list=[
                ('pipeline_categoric', sklearn.pipeline.Pipeline(steps=[
                   ('select_columns',      sklearn.preprocessing.FunctionTransformer(self.subcolumns, validate=False, kwargs='auto')),  # {'indices': 'data__categoric_ind_name'}
                   ('encode_onehot',       mlshell.custom.SkippableOneHotEncoder(handle_unknown='ignore', categories='auto', sparse=False, drop=None, skip=False)),
                ])),
                ('pipeline_numeric',   sklearn.pipeline.Pipeline(steps=[
                    ('select_columns',     sklearn.preprocessing.FunctionTransformer(self.subcolumns, validate=False, kwargs='auto')),  # {'indices':  'data__numeric_ind_name'}
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
            ('select_columns',   sklearn.feature_selection.SelectFromModel(estimator=mlshell.custom.CustomSelectorEstimator(estimator_type=estimator_type, verbose=False, skip=True), prefit=False)),
            ('reduce_dimension', mlshell.custom.CustomReducer(skip=True)),
            ('estimate', self.last_step(estimator, estimator_type, th_strategy=th_strategy)),
        ]

    def last_step(self, estimator, estimator_type, th_strategy=None):
        if estimator_type == 'regressor':
            last_step = sklearn.compose.TransformedTargetRegressor(regressor=estimator, transformer=None, check_inverse=True)
        elif estimator_type == 'classifier':
            if th_strategy == 0 or not th_strategy:
                last_step = sklearn.pipeline.Pipeline(steps=[('classifier', estimator)])
            else:
                last_step = sklearn.pipeline.Pipeline(steps=[
                        ('predict_proba',   mlshell.custom.PredictionTransformer(estimator)),
                        ('apply_threshold', mlshell.custom.ThresholdClassifier(threshold=0.5,
                                                                               kwargs='auto')),
                        ])
        else:
            raise ValueError(f"Unknown estimator type `{estimator_type}`.")

        if sklearn.base.is_classifier(estimator=last_step) ^ (estimator_type == "classifier"):
            raise MyException('MyError:{}:{}: wrong estimator type'.format(self.__class__.__name__,
                                                                           inspect.stack()[0][3]))
        return last_step

    def get_steps(self):
        """Pipeline steps getter."""
        return self._steps

    def set_scorer_kwargs(self, *args, **kwargs):
        """Mock function to allow set custom kwargs."""
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


# Nuances
#   user can set 'auto' or ['auto',] for hps need to resolve in resolver (for some hps is default).
#   user can skip unify step, then nummeric/categoric_ind_names auto extracted in resolver.
#       it`s possible to move out in DataFactory like mandatory, but users not always need this, only for resolving.
#   target_names, categor_names, pos_labels should be alwaus list or None

# My Hints:
#   * we need to find compromise between user interface and keep close code in one block.
#   * if we want key error not use get, just use [].
#       get don`t give additional comliance to between dict keys and attributes.
#   * arr = dic.get('arr',None)
#       "if not arr" give error "The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()".
#       "if arr is not None" is better.