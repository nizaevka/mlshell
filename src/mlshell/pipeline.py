"""
TODO:
    load pipeline will not work, because it don`t know about input chanel
    visualize pipeline

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


def set_base(pipeline):
    class Pipeline(pipeline):
        def fit(self):
            self.pipeline.fit()
            return self

        def optimize(self):
            pass

        def validate(self):
            pass

        def predict(self):
            pass

        def dump(self):
            pass

    return Pipeline


class PipeFactory(object):
    def __init__(self, project_path, logger=None):
        super().__init__(project_path, logger)
        if logger is None:
            self.logger = logging.Logger('PipeFactory')
        else:
            self.logger = logger
        self.project_path = project_path

    def produce(self, pipeline_id, p):
        """ Create/load pipeline in compliance to workflow class.

        Arg:
            p():
        Note:

        """
        self.logger.info("\u25CF HANDLE PIPELINE")
        self.logger.info(f"Pipeline configuration:\n    {pipeline_id}")

        if 'load' in p and 'filepath' in p['load'] and isinstance(p['load']['filepath'], str):
            pipeline = self.load(**p['load'])
        else:
            pipeline = self.create(**p['create'])

        return pipeline  # [deprecated] set_base(pipeline)

    def load(self, filepath=None, **kwargs):
        """Load fitted model on disk/string.

        Note:
            Better use only the same version of sklearn.

        """
        self.logger.info("\u25CF LOAD PIPELINE")
        pipeline = joblib.load(filepath)
        self.logger.info('Load fitted model from file:\n    {}'.format(filepath))
        return pipeline

    def create(self, **kwargs):
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
        pipeline_ = self._pipeline_steps(**kwargs)

        last_step = self._create_last(kwargs['estimator'], pipeline_, kwargs['type'])
        self.logger.info(f"Estimator step:\n    {last_step}")
        pipeline_.append(('estimate', last_step))
        pipeline = sklearn.pipeline.Pipeline(pipeline_, memory=cachedir)
        # run tests
        sklearn.utils.estimator_checks.check_estimator(pipeline, generate_only=False)

        return pipeline

    def _create_last(self, estimator, pipeline, estimator_type):
        """Create last step of pipeline

        Args:
            estimator (sklearn estimator object): to use in last step
            pipeline_ (list of pipeline steps):will use repack 'estimate' for regression
            estimator_type: (str): 'regressor' or 'classifier'
            input: {dict}: storage to dynamically pass dataset`s attributes in pipeline .

        Returns:
            last_step (pipeline object): last_step

        Note:
            if regression: will use 'estimate' if provided
            if classification: will raise error 'estimate', add custom threshold tuner

        TODO:
            should be user defined, self.classes user-defined or auto
        """



        return last_step

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


