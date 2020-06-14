"""
Pipeline should contain:

fit():
    .fit()
    .set_params()
    .get_params()
optimize():
    sklearn estimator
validate():
    predict_proba()/predict()
"""


from mlshell.libs import *
import mlshell


class PipeProducer(mlshell.Producer):
    def __init__(self, project_path='', logger=None):
        self.logger = logger if logger else logging.Logger(__class__.__name__)
        self.project_path = project_path
        super().__init__(self.project_path, self.logger)

    def load(self, pipeline, filepath=None, **kwargs):
        """Load fitted model on disk/string.

        Note:
            Better use only the same version of sklearn.

        """
        self.logger.info("\u25CF LOAD PIPELINE")
        pipeline.pipeline = joblib.load(filepath)
        self.logger.info('Load fitted model from file:\n    {}'.format(filepath))
        return pipeline

    def create(self, pipeline, **kwargs):
        """  Create pipeline

        Note:
            it is possible to use cache
                cache each transformer after calling fit
                avoid double calculation of transformers in GridSearch
                will use cache result if steps and params are the same
            but error-prone (better user-level control)
                https://scikit-learn.org/stable/modules/compose.html#caching-transformers-avoid-repeated-computation
                https://github.com/scikit-learn/scikit-learn/issues/10068
                bad-tested
                will be problem in case of transformer is changed internally
                giant hdd consuming
                time consuming create hash from GB of dat

        """
        self.logger.info("\u25CF CREATE PIPELINE")
        if kwargs['cache']:
            cachedir = f"{self.project_path}/results/cache/pipeline"
            # delete cache if necessary
            if kwargs['cache'] == 'update' and os.path.exists(cachedir):
                shutil.rmtree(cachedir, ignore_errors=True)
                self.logger.warning(f'Warning: update pipeline cache:\n    {cachedir}')
            else:
                self.logger.warning(f'Warning: pipeline use cache:\n    {cachedir}')
            if not os.path.exists(cachedir):
                # create temp dir for cache if not exist
                os.makedirs(cachedir)
        else:
            cachedir = None

        # assemble several steps that can be cross-validated together
        steps = self._pipeline_steps(**kwargs)

        # [deprecated]
        # last_step = self._create_last(kwargs['estimator'], pipeline_, kwargs['estimator_type'])
        # self.logger.info(f"Estimator step:\n    {last_step}")
        # pipeline_.append(('estimate', last_step))

        pipeline.pipeline = sklearn.pipeline.Pipeline(steps, memory=cachedir)
        # run tests
        # sklearn.utils.estimator_checks.check_estimator(pipeline, generate_only=False)

        return pipeline

    def _pipeline_steps(self, steps=None, **kwargs):
        """Configure pipeline steps.

        Returns:
            sklearn.pipeline.Pipeline steps.

        Note:
            * | feature/object selections should be independent at every fold,
              | otherwise bias (but ok if totally sure no new data)
            * can`t pass params to fit() in case of TargetTransformation
            * if validate=True in FunctionTransformer, will raise error on "np.nan"
            * limitation of pickling (used for cache and parallel runs on pipeline).

                * define custom function in a module you import, or at least not in a closure.
                * | pickle can`t pickle lambda, cause python pickles by name reference, and a lambda doesn't have a name.
                  | preprocessing.FunctionTransformer(
                  | lambda data: self.subcolumns(data, self.categoric_ind_name),validate=False))

        TODO:
            use dill instead.

        """
        if isinstance(steps, list):
            steps = steps
            self.logger.warning('Warning: user-defined pipeline is used instead of default.')
        else:
            if steps is None:
                clss = mlshell.default.CreateDefaultPipeline
            else:
                clss = steps
                self.logger.warning('Warning: user-defined pipeline is used instead of default.')
            steps = clss(**kwargs).get_steps()
        return steps


# [deprecated] not work
# def set_base(pipeline):
#     class Pipeline(pipeline):
#   return Pipeline


class Pipeline(object):
    def __init__(self, pipeline=None):
        """

        Attributes:
            pipeline:
                Object for which wrapper is created.
        """
        # not sure if self.fit will change self.pipeline [bad practice i think]
        self.pipeline = pipeline
        # Need only to add dataset_id to name for fitted pipeline when dump on disk.
        # Can be skipped.
        # TODO [code-review]: maybe better internal storage(actuaaly can do both)
        self.dataset_id = None

        # [deprecated] move out to optimizer
        # self.optimizers = []
        # self.best_params_ = {}
        # self.best_score_ = float("-inf")

    def __getattr__(self, name):
        """Redirect unknown methods to pipeline object."""
        def wrapper(*args, **kwargs):
            getattr(self.pipeline, name)(*args, **kwargs)
        return wrapper

    def __hash__(self):
        return str(self.pipeline.get_params())

    def fit(self, *args, **kwargs):
        self.dataset_id = kwargs.pop('dataset_id', None)
        self.pipeline.fit(*args, **kwargs)

    def set_params(self, *args, **kwargs):
        self.dataset_id = None
        self.pipeline.set_params(*args, **kwargs)

    def is_classifier(self):
        return sklearn.base.is_classifier(self.pipeline)

    def is_regressor(self):
        return sklearn.base.is_regressor(self.pipeline)

    def resolve(self, hp_name, dataset, **kwargs):
        resolver = kwargs.get('resolve', {}).get(hp_name, {}).get('resolver', mlshell.HpResolver)
        # [deprecated] need all level kwargs
        # sub_kwargs = kwargs.get('resolve', {}).get(hp_name, {})
        return resolver().resolve(self.pipeline, dataset, hp_name, **kwargs)

    def dump(self, file):
        joblib.dump(self.pipeline, file)
        return

# [deprecated] move out
#    def update_params(self, optimizer):
#        """
#        Note:
#            work good for refine runs, could lose runs if the same param multiple times
#            in thar case need to merge 'cv_results_' and recalc best each times
#            but could be different number of split, hard to combine different optimizers
#
#            # best_params_ (not available if refit is False and multi-metric)
#            # best_estimator availbale if refit is not False
#            # best_score available if refit is not False and not callable
#
#            # TODO: discuss alternative storage pipeline as optimizer object and best_* attributes
#            #       more logically and sklearn-consistent, but less flexible for different optimizers
#            #       best_estimator not always available to fit
#            # Optimizer fit it is not the same as pipeline fit, it is better
#        """
#        self.pipeline = optimizer.__dict__.get('best_estimator_', pipeline)
#        self.best_params_.update(optimizer.__dict__.get('best_params_', {}))
#
#        # [alternative] not evailable if refit callable
#        # best_score_ = optimizer.__dict__.get('best_score_', float("-inf"))
#        # if pipeline.best_score_ <= best_score_:
#        #        self.pipelines[pipeline_id].best_score_ = best_score_
#
#    # [deprecated] need to use in fit(), predict() ..
#    # def ckeck_data_format(self, data):
#    #     """additional check for data when pass to pipeline."""
#    #     if not isinstance(data, pd.DataFrame):
#    #         raise TypeError("input data should be pandas.DataFrame object")
#    #     return
