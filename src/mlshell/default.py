"""Module contains default configuration and pipeline steps.

Workflow configuration is set with python dictionary.
It could be passed to `mlshell.run()` function or some user-defined handler,
where workflow class is built and its endpoints are executed.

There is common logic for all configuration sections:
{'section_id':
    'configuration_id 1': {
        'init': initial object of custom type.
        'producer': factory class, which contain methods to run steps.
        'patch': add custom method to class.
        'steps': [
            ('method_id 1', kwargs_1),
            ('method_id 2', kwargs_2),
        ],
        'global': shortcut to common parameters.
        'priority': execute priority (integer non-negative number).
    }
    'configuration_id 2':{
        ...
    }
}
The target for each section is to create object (pipeline, dataset, ..).
`producer` object work as factory, it should contain .produce() method which:
* takes `init` consecutive pass it to `steps` with additional kwargs.
* return resulting object.
so `init` is the template for produced object (empty dict() for example).

Created `objects` can be used as kwargs for any step in others sections.
But the order in which sections handled is important. For this purpose
'priority' key is available: non-negative integer number, the more the higher
the priority (by default all set to 0). For two with same priority order is
not guaranteed.

`mlshell.run()` handler:
* Parse section one by one in priority.
* For each configuration in sections:
    * call `producer`.produce(`init`, `steps`, `objects`).
    * store result in built-in `objects` storage under `section_id__configuration_id`.

For flexibility, it is possible to:
* monkey patch `producer` object with custom functions via `patch` key.
* specify global value for common kwargs in steps via `global` key.
* create separate section for any configuration subkey or kw_arg parameter in
steps. TODO: any configuration subkeys + need rearange read conf.
there are two ways:
    * use `section_id` to deepcopy target configuration `init` before execute steps.
    * use `section_id__id` postfix to pass `configuration_id` as kwargs.
* TODO: skip anykwargs in steps, will be used class default.
    actually default better specify in class functions.

For ML task, common sections would be:
* create/read pipelines and datasets objects.
* create workflow class and call methods with pipeline/dataset as argument.

# TODO: do both producer as class or object.
#    better give flexibility for user
#    and in default i can use class.

# TODO:
#    encompass all sklearn-wise in mlshell.utills.sklearn

# TODO:
    kwargs(not kwargs) everywhere in documentation/code

See default configuration for example.
"""


from mlshell.libs import *
import mlshell.custom
import mlshell.optimize
import mlshell.validate


WORKFLOW = {
    'default': {
        'init': {},
        'producer': mlshell.Workflow(project_path=project_path, logger=logger),
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
        'init': mlshell.Pipeline(),
        'producer': mlshell.PipeProducer(project_path=project_path, logger=logger),
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
        'producer': mlshell.MetricProducer(project_path=project_path, logger=logger),
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
        'producer': mlshell.MetricProducer(project_path=project_path, logger=logger),
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
        'class': mlshell.DataProducer(project_path=project_path, logger=logger),
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
""" Default sections for ML task.

Each section specify set of configurations.
Each configuration provide steps to construct an object, that can be
utilize as argument in some other sections.
See below detailed configuration keys description.

TODO: Better move to producer.
    Default better move to read conf.

Parameters
----------
init : object.
    Initial state for constructed object. Will be passed consecutive in steps
    as argument.
    If None or skipped, dict() is used.
    
producer : class or instance.
    Factory to construct object, producer.produce(`init`, `steps`, `objects`) 
    will be called, where `objects` is dictionary with previously created 
    objects {'section_id__configuration_id': object}.  
    TODO: 
    If None or skipped, mlshell.Producer is used.
    If class will be initialized with producer(project_path, logger).
    
patch : dict {'method_id' : function}.
    Monkey-patching `producer` object with custom functions.
    
steps : list of tuples (str: 'method_id', Dict: kwargs).
    List of class methods to run consecutive with kwargs. 
    Each step should be a tuple: `('method_id', {kwargs to use})`,
    where 'method_id' should match to `producer` functions' names.
    It is possible to omit kwargs, in that case each step executed with kwargs
    set default in corresponding producer method (see producer interface)

    **kwargs : dict {'kwarg_name': value, ...}.
        Arguments depends on workflow methods. It is possible to create
        separate configuration section for any argument. If value is set here
        to None, parser try to resolve it. First it searches for value in
        `global` subsection. Then resolver looks up 'kwarg_name' in section
        names. If such section exist, there are two possibilities: if
        `kwarg_name` contain '_id' postfix, resolver substitutes None with
        available `configuration_id`, else without postfix
        resolver substitutes None with copy of configuration `init` object.
        If fails to find resolution, value is remained None. In case of plurality of 
        resolutions, ValueError is raised.
        TODO: check if?
        Also list of id is possible (like for metric)

global : dict {'kwarg_name': value, ...}.
    Specify values to resolve None for arbitrary kwargs. This is convenient for
    example when we use the same `pipeline` in all methods. It is not rewrite 
    not-None values.


Examples
--------
# Use custom functions.
def my_func(self, pipeline, dataset):
    # ... custom logic ...
    return 

{'patch': {'extra': my_func,},}

Notes
-----
If section is skipped, default section is used.
If sub-keys are skipped and `producer` is None/skipeed, default values are used 
for these sub-keys. So in most cases it is enough just to specify 
'global'.

See also
--------
:class:`Workflow`:
    default universal workflow class.

"""
"""

endpoint : str or None.
    Endpoint identifier, should be the one described in `endpoint section`.
    Auto-resolved if None and endpoint section contain only one configuration,
    else ValueError.
    TODO: [beta].
    It`s possible to specify list of endpoints to run consecutive.
    In that case 'steps' should be list of lists.

steps : list of tuples (str 'step_identifier', kwargs).
    List of workflow methods to run consecutive. 
    Each step should be a tuple: `('step_identifier', {kwargs to use})`,
    where 'step_identifier' should match with `endpoint` functions' names.
    By default, step executed with argument taken from `endpoint section`,
    but it also is possible to update kwargs here before calling.
    
 

Notes
-----
If section is skipped, default template is used.
If subkeys are skipped, default values are used for these subkeys.

See also
--------

"""

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