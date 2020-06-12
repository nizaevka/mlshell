"""Module contains default workflow configuration and pipeline steps.

Workflow configuration is set with python dictionary.
It could be passed to the `mlshell.run()` function or some user-defined handler, where
workflow class is built and its endpoints are executed.

There is common logic for all configuration sections:
{'section_id':
    'configuration_id 1': {
        'init': initial object of custom type.
        'producer': factory class, which contain methods to run steps.
        'patch': add custom method to class.
        'steps': [
            ('method_id 1', kw_args_1),
            ('method_id 2', kw_args_2),
        ],
        'global': shortcut to common parameters.
        'priority': parsing priority (integer number).
    }
    'configuration_id 2':{
        ...
    }
}
The target of each section is to create object (pipeline, dataset, ..).
`producer` work as factory, it should contain .produce() method, which is:
* takes `init` and consecutive pass it to `steps` with additional kwargs.
* return resulting object.
so `init` is the future object template (empty dict() for example).

Created objects can be used as kw_args for any step in others sections or even
as `init`/`producer` object. But it is important the order in which
sections handled. For this 'priority' key is available, otherwise default
python dict() keys order is used.

`mlshell.run()` handler:
* Parse section one by one in priority.
* For each configuration in section:
    * call .produce(`init`, `steps`) on `producer`.
    * store result in internally storage under `section_id__configuration_id`.

For flexibility, it is possible to:
* monkey patch `producer` object with custom functions via `patch` key.
* specify global value for common kw_args in steps via `global` key.
* create separate section for any configuration`s subkey or kw_arg parameter.

For ML task, common sections would be:
* create/read pipelines and datasets objects.
* create workflow class and call methods with pipeline/dataset as argument.

"""


from mlshell.libs import *
import mlshell.custom
import mlshell.optimize
import mlshell.validate


TODO: move logger, find_path in conf.py as class creation argument

WORKFLOW = {
    # DONE
    #     when need fit two times
    #        * multiplpe endpoints [best]
    #        * either different names 'fit2': {'func':'fit'}  [better, cause rare situation]
    #        * either additional argument in steps [not beautiful]
    #     DO BOTH!
    #     possibility of multiple pipelines, metrics lists, gs_conf in one workflow!
    #     in second not possible to change func

    # TODO [beta]: Data/Pipeline/Workflow: endpoint(class+func to replace), kwargs, steps.
    #    Also need class for GUI.
    #    but it will be less beautiful config.
    #    Remain current change if needed. Maybe erson don`t want to use func.
    'endpoint_id': None,    # TODO: move out id or everywhere set id? I think better name_id,
                            #     so additional explicit flag to search in config, but then "gs_params_id" not so good
                            #     вообще итак нормально.
    'steps': [
        ('fit',),
        ('optimize',),
        ('dump',),
        ('validate',),
        ('plot',),
        ('predict',),
    ],
}
""" Workflow section.

Specify endpoint configuration to construct workflow class.
Specify the order of workflow methods to execute.

Parameters
----------
endpoint : str or None.
    Endpoint identifier, should be the one described in `endpoint section`.
    Auto-resolved if None and endpoint section contain only one configuration,
    else ValueError.
    TODO: [beta].
    It`s possible to specify list of endpoints to run consecutive.
    In that case 'steps' should be list of lists.
    
steps : list of tuples (str 'step_identifier', kw_args).
    List of workflow methods to run consecutive. 
    Each step should be a tuple: `('step_identifier', {kw_args to use})`,
    where 'step_identifier' should match with `endpoint` functions' names.
    By default, step executed with argument taken from `endpoint section`,
    but it also is possible to update kw_args here before calling. 

Notes
-----
If section is skipped, default template is used.
If subkeys are skipped, default values are used for these subkeys.

See also
--------

"""


ENDPOINTS = {
    'default': {
        'class': None,
        'global': {},
        'steps': [],
        
        'dump': {'func': None,
                 'pipeline_id': None,},
        # both
        'optimize': {'func': None,
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
                     },
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
                'pipeline_id': None,
                'dataset_id': 'train',
                'fit_params': {},
                'hp': {},
                'resolve_params': {},
                },
        'validate': {'func': None,
                     'dataset_id': 'train',
                     'validator': None,
                     'metric': None,
                     'pos_label': None,  # if None, get -1
                     'pipeline_id': None,},
        'predict': {'func': None,
                    'dataset_id': 'test',
                    'pipeline_id': None,},
        'plot': {'func': None,
                'plotter': None,  # gui
                'pipeline_id': None,
                'hp_grid': {},
                'dataset_id': 'train',
                'base_sort': False,
                # TODO: [beta]
                # 'dynamic_metric': None,
                },
        # TODO: [beta] memory
        # 'init': {},
        # 'reset': {},
    },
    # TODO: [beta]
    # 'pytorch': {},
}
""" Endpoint section.

Specify separate endpoints configurations to fast switch between different
workflow classes {'endpoint_identifier': parameters, ...}

Parameters
----------
class : class or None.
    TODO !!!! always should be in endpoint, factory class 
    The class used to construct workflow instance.
    If None, default Workflow class is used.

**methods : dict {'method_name': {'func': None, **kwargs}, ...}.
    Each key corresponds to one workflow method. Each value specifies kw_args 
    to call that method . Special subkey `func` used to reassign target
    function if needed. It takes either string name of `producer` method, None, or
    custom function. If `func` is None, default `producer` method is used. 
    
    **kwargs : dict {'kwarg_name': value, ...}.
        Arguments depends on workflow methods. It is possible to create
        separate configuration section for any argument. If value is set here
        to None, parser try to resolve it. First it searches for value in
        `global` subsection. Then resolver looks up 'kwarg_name' in section
        names. If such section exist, there are two possibilities: if
        `kwarg_name` contain '_id' postfix, resolver substitutes None with
        existed id of available configuration in section, else without postfix
        resolver substitutes None with configuration itself. If case of fail
        to find resolution, value is remained None. In case of plurality of 
        resolutions, ValueError is raised.
        TOOO !!!!
        configuration with postfix will be passed to workflow separately.
        Also list of id is possible (like for metric)
        
global : dict {'kwarg_name': value, ...}.
    Specify values to resolve None for arbitrary kwargs. This is convenient for
    example when we use the same `pipeline` in all methods. Doesnt't rewrite 
    not-None values.
    
steps : list of tuples (str 'step_identifier', kw_args).
    List of class methods to run consecutive. 
    Each step should be a tuple: `('step_identifier', {kw_args to use})`,
    where 'step_identifier' should match with `producer` functions' names.
    By default, step executed with argument taken from **methods,
    but it is also possible to update kw_args before calling. 
        

Examples
--------
# Use built-in class methods` names to specify separate kw_args configuration.
'dump_1': {'func': 'dump',
            'pipeline_id': 'pipeline_1',},
'dump_2': {'func': 'dump',
          'pipeline_id': 'pipeline_2',},

# Use custom functions.
def my_func(self, pipeline, dataset):
    # custom logic
    return 
'custom': {'func': my_func,
           'pipeline_id': 'xgboost',
           'dataset_id': 'train'},
    
Notes
-----
If section is skipped, default endpoint is added.
Otherwise for each endpoint if `producer` is None or not set, default values are
used for skipped subkeys. So in most cases it is enough just to specify 
'global' subsection.

See also
--------
:class:`Workflow`:
    default universal workflow class.

"""

# TODO: if class is provided => Factory with produce method => workflow via argument
#       else =>  => workflow via argument without changes

PIPELINES = {
    'default': {
        'class': None,  # Factory

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
}


METRICS = {
    'classifier': {
        'class': None,
        'func':sklearn.metrics.accuracy_score,
        'kw_args': {'greater_is_better': True},
    },
    'regressor': (sklearn.metrics.r2_score, {'greater_is_better': True}),
}


DATASETS = {
    'default': {
        'class': None,  # Factory
        # TODO: both steps and reaassign
        'steps': [
            ('load_cache', {'func': None, 'prefix': None},),
            ('get', {'func': None},),
            ('preprocess', {'func': None, 'categor_names': [], 'target_names': [], 'pos_labels': []},),
            ('info', {'func': None},),
            ('unify', {'func': None},),
            ('split', False,),
            ('dump_cache', {'func': None, 'prefix': None},),
        ],
    },
}


CUSTOMS = {
    'gs_params': {
        'default': {
            'hp_grid': {},
            'n_iter': None,  # ! my resolving (1, hp_grid number), otherwise 'NoneType' object cannot be interpreted as an integer
            'scoring': None,  # no resolving (default estimator scoring)
            'n_jobs': 1,
            'refit': None, # no resolving
            'cv': sklearn.model_selection.KFold(n_splits=3, shuffle=True),
            'verbose': 1,
            'pre_dispatch': 'n_jobs',
        },
    },
}


DEFAULT_PARAMS = {
    'workflow': WORKFLOW,
    'endpoint': ENDPOINTS,
    'pipeline': PIPELINES,
    'dataset': DATASETS,
    'metric': METRICS,
    **CUSTOMS,
}

# TODO:
#    encompass all sklearn-wise in mlshell.utills.sklearn


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