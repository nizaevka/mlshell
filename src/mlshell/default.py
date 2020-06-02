"""Module contains default configuration for user params and class to create default pipeline steps."""


from mlshell.libs import *
import mlshell.custom
import mlshell.optimize
import mlshell.validate


# TODO:  there are asymmetry pipelines, datasets are separate object, but what if gs_params config?
#  does it need fabric? do we need self.p? maybe it is possible to combine all?
#  DO it in the end

DEFAULT_PARAMS = {
    # when need fit two times
    #    * multiplpe endpoints [best]
    #    * either different names 'fit2': {'func':'fit'}  [better, cause rare situation]
    #    * either additional argument in steps [not beautiful]
    # DO BOTH!
    # possibility of multiple pipelines, metrics lists, gs_conf in one workflow!
    # in second not possible to change func

    # TODO [beta]: Data/Pipeline/Workflow: endpoint(class+func to replace), kwargs, steps.
    #    Also need class for GUI.
    #    but it will be less beautiful config.
    #    Remain current change if needed. Maybe erson don`t want to use func.
    'workflow': {
        'endpoint_id': None,    # TODO: move out id or everywhere set id? I think better name_id,
                                #     so additional explicit flag to search in config, but then "gs_params_id" not so good
                                #     вообще итак нормально.
        'steps': [
            ('fit',),
            # ('fit', 0, {'pipeline':'pipeline_2'}),
            ('optimize',),
            # ('optimize', 1, {'optimizer': None, 'gs_params':'gs_2', 'th_name': 'estimate__apply_threshold__threshold'})
            ('dump',),
            ('validate',),
            ('gui',),
            ('predict',),
        ],
    },
    'endpoint': {
        # so workflow class have set of built-in function, so we can resolve None by key name
        # if fit_2 can`t resolve => set from built in or new

        'default': {
            # only second-level keys copy if skipped, not third
            # 'global': {
            #    'class': None,
            #    'seed': None,
            #     'pipeline': None,
            #     'dataset': None,
            #     'metric': None,
            #     'gs': None,
            # },
            # [deprecated] not necessary, better on call and del
            ## data
            #'handle': {},
            # pipeline
            ##'add': {},


            ## call pipeline endpoints
            # no data
            # 'load': {'func': None,
            #          'pipeline': None,
            #          'seed': None},
            # 'load2': {'func': 'load',
            #          'pipeline': None,
            #          'seed': None},
            # 'create': {'func': None,
            #            'pipeline': None,
            #            'seed': None},
            'dump': {'func': None,
                     'pipeline': None,
                     'seed': None},
            'dump2': {'func': 'dump',
                      'pipeline': None,
                      'seed': None},
            # both
            'optimize': {'func': None,
                         'optimizer': mlshell.optimize.RandomizedSearchOptimizer,  # optimizer
                         'validator': mlshell.validate.Validator,
                         'pipeline': None,  # multiple pipeline? no, user can defined separately if really needed
                         'dataset': 'train',
                         'gs_params': None,
                         'fit_params': {},
                         'resolve_params': {
                             'estimate__apply_threshold__threshold': {
                                 'resolver': None,  # if None, use pipeline default resolver
                                 'samples': 10,
                                 'plot_flag': False,
                             },
                         },
                         },
            'fit': {'func': None,
                    'pipeline': None,
                    'dataset': 'train',
                    'fit_params': {},
                    'hp': {},
                    'seed': None,
                    'resolve_params': {},
                    },
            'validate': {'func': None,
                         'dataset': 'train',
                         'validator': None,
                         'metric': None,
                         'pos_label': None,  # if None, get -1
                         'pipeline': None,
                         'seed': None},
            'predict': {'func': None,
                        'dataset': 'test',
                        'pipeline': None,
                        'seed': None},
            'gui': {'plotter': None, # gui
                    'pipeline': None,
                    'hp_grid': {},
                    'dataset': 'train',
                    'base_sort': False,
                    # 'metric': False,  beta
                    'seed': None},
            # memory
            'init': {},
            'reset': {},

        },
    },
    'pipeline': {
        'default': {
            'class': None,  # Factory

            # one of two
            'load': {'func': None,
                     'filepath': None,
                     'estimator_type': 'regressor',
                     },
            'create': {'func': None,
                       'cache': None,
                       'steps': None,
                       'estimator': sklearn.linear_model.LinearRegression(),
                       'estimator_type': 'regressor',
                       # 'th_strategy': None,
                       },
            'resolve': {
                 'func': None,
                 # [deprecated]  should be setted 'auto'/['auto'], by default only for index
                 #  only if not setted
                 # 'hp': {
                 #     'process_parallel__pipeline_categoric__select_columns__kw_args',
                 #     'process_parallel__pipeline_numeric__select_columns__kw_args',
                 #     'estimate__apply_threshold__threshold'}
                 # },
            },
        },
    },
    'metric': {
        # 'default': (sklearn.metrics.r2_score, {'greater_is_better': True}),
        'classifier': (sklearn.metrics.accuracy_score, {'greater_is_better': True}),
        'regressor': (sklearn.metrics.r2_score, {'greater_is_better': True}),
    },
    # can be, but not necessary
    # 'optimizer': {
    #     'default': {
    #         'class': None,

    #     }
    # },
    'gs_params': {
        'default': {
            # 'flag': True,
            'hp_grid': {},
            'n_iter': None,  # ! my resolving (1, hp_grid number), otherwise 'NoneType' object cannot be interpreted as an integer
            'scoring': None,  # no resolving (default estimator scoring)
            'n_jobs': 1,
            'refit': None, # no resolving
            'cv': sklearn.model_selection.KFold(n_splits=3, shuffle=True),
            'verbose': 1,
            'pre_dispatch': 'n_jobs',
            # not necessary to specify all
            #'random_state': None,
            #'error_score': np.nan,
            #'return_train_score': True,
        },
        'gs_2': {
            'hp_grid': {'threshold': [0.5]},
        },
    },
    # TODO: test that we can set as configuration
    # 'resolve_params':{},
    'dataset': {
        'default': {
            'class': None,  # Factory
            'steps': [
                ('load_cache', {'flag': 0, 'func': None, 'prefix': None},),
                ('get', {'func': None},),
                ('preprocess', {'func': None, 'categor_names': [], 'target_names': [], 'pos_labels': []},),
                ('info', {'func': None},),
                ('unify', {'func': None},),
                ('split', False,),
                ('dump_cache', {'flag': 0, 'func': None, 'prefix': None},),
            ],
        },
    },
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
            ('pass_custom',      sklearn.preprocessing.FunctionTransformer(func=self.set_scorer_kw_args, validate=False)),
            ('select_rows',      sklearn.preprocessing.FunctionTransformer(func=self.subrows, validate=False)),
            ('process_parallel', sklearn.pipeline.FeatureUnion(transformer_list=[
                ('pipeline_categoric', sklearn.pipeline.Pipeline(steps=[
                   ('select_columns',      sklearn.preprocessing.FunctionTransformer(self.subcolumns, validate=False, kw_args='auto')),  # {'indices': 'data__categoric_ind_name'}
                   ('encode_onehot',       mlshell.custom.SkippableOneHotEncoder(handle_unknown='ignore', categories='auto', sparse=False, drop=None, skip=False)),
                ])),
                ('pipeline_numeric',   sklearn.pipeline.Pipeline(steps=[
                    ('select_columns',     sklearn.preprocessing.FunctionTransformer(self.subcolumns, validate=False, kw_args='auto')),  # {'indices':  'data__numeric_ind_name'}
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
                                                                               kw_args='auto')),
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

    def set_scorer_kw_args(self, *args, **kw_args):
        """Mock function to allow set custom kw_args."""
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