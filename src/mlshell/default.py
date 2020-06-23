"""The module contains default configuration and pipeline steps."""


from mlshell.libs import *
import mlshell.custom
import mlshell.optimize
import mlshell.validate

__all__ = ['DEFAULT_PARAMS', 'PipelineSteps']


PATHS = {
    'default': {
        'init': pycnfg.find_path,
        'producer': pycnfg.Producer,
        'global': {},
        'patch': {},
        'priority': 1,
        'steps': [],
    }
}


LOGGERS = {
    'default': {
        'init': 'default',
        'producer': mlshell.LoggerProducer,
        'global': {},
        'patch': {},
        'priority': 2,
        'steps': [
            ('create', {})
        ],
    }
}


PIPELINES = {
    'default': {
        'init': mlshell.Pipeline,
        'producer': mlshell.PipeProducer,
        'global': {},
        'patch': {},
        'priority': 3,
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
        'priority': 3,
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
        'priority': 3,
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
        'producer': mlshell.DataProducer,
        'global': {},
        'patch': {},
        'priority': 3,
        'steps': [
            ('load_cache', {'prefix': None},),
            ('load', {},),
            ('preprocess', {'categor_names': [], 'target_names': [], 'pos_labels': []},),
            ('info', {},),
            ('unify', {},),
            ('split', {},),
            ('dump_cache', {'prefix': None},),
        ],
    },
}


WORKFLOWS = {
    'default': {
        'init': {},
        'producer': mlshell.Workflow,
        'global': {},
        'patch': {},
        'priority': 4,
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
                        'resolver': None,
                        'samples': 10,
                        'plot_flag': False,
                        'fit_params': {},
                        'cv': sklearn.model_selection.KFold(n_splits=3, shuffle=True),
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


DEFAULT_PARAMS = {
    'path': PATHS,
    'logger': LOGGERS,
    'pipeline': PIPELINES,
    'dataset': DATASETS,
    'metric': METRICS,
    'workflow': WORKFLOWS,
}
"""Default sections for ML task.

For ML task, common sections would be:
* create/read pipelines and datasets objects.
* create workflow class and call methods with pipeline/dataset as argument.

"""


@nb.njit
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


class PipelineSteps(object):
    """Class to create pipeline steps.

        Parameters
        ----------
        estimator : object with sklearn.pipeline.Pipeline interface
            Estimator to use in the last step.
            If `estimator_type`='regressor':
            sklearn.compose.TransformedTargetRegressor(regressor=`estimator`)
            If `estimator_type`='classifier' and `th_step`=True:
            sklearn.pipeline.Pipeline(steps=[
                ('predict_proba',
                    mlshell.custom.PredictionTransformer(`estimator`)),
                ('apply_threshold',
                    mlshell.custom.ThresholdClassifier(threshold=0.5,
                                                       kwargs='auto')),
                        ])
            If `estimator_type`='classifier' and `th_step`=False:
            sklearn.pipeline.Pipeline(steps=[('classifier', `estimator`)])
        estimator_type : str
            'estimator` or 'regressor'.
        th_step : bool
            If True and 'classifier', ddd `mlshell.custom.ThresholdClassifier`
            sub-step. Otherwise ignored.

        Attributes
        ----------
        steps : list
            Pipeline steps to pass in `sklearn.pipeline.Pipeline`.

        Notes
        -----
        Assemble steps in class are made for convenience.
        By default, 4 parameters await for resolution:
            'process_parallel__pipeline_categoric__select_columns__kwargs'
            'process_parallel__pipeline_numeric__select_columns__kwargs'
            'estimate__apply_threshold__threshold'
            'estimate__apply_threshold__kwargs'

        'pass_custom' step should be the first or absent.
        'estimate' step should be the last, by default:

    """
    _required_parameters = ['estimator', 'estimator_type']

    def __init__(self, estimator, estimator_type, th_step=False):
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
            ('estimate', self.last_step(estimator, estimator_type, th_step=th_step)),
        ]

    def last_step(self, estimator, estimator_type, th_step):
        """Prepare estimator step."""
        if estimator_type == 'regressor':
            last_step = \
                sklearn.compose.TransformedTargetRegressor(regressor=estimator)
        elif estimator_type == 'classifier' and th_step:
            last_step = sklearn.pipeline.Pipeline(steps=[
                ('predict_proba',
                    mlshell.custom.PredictionTransformer(estimator)),
                ('apply_threshold',
                    mlshell.custom.ThresholdClassifier(threshold=0.5,
                                                       kwargs='auto')),
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
        """Pipeline steps getter."""
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


if __name__ == '__main__':
    pass
