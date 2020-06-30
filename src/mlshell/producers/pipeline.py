""""
The :mod:`mlshell.pipeline` contains examples of `Pipeline` class to create
empty pipeline object and `PipelineProducer` class to fulfill it.

`Pipeline` class proposes unified interface to work with underlying pipeline.
Intended to be used in `mlshell.Workflow`. For new pipeline formats no need to
edit `Workflow` class, only update `Pipeline` interface logic.

`PipelineProducer` class specifies methods to create/load pipeline.
Current implementation provides sklearn.pipeline.Pipeline(steps) model creation
via steps and model loading via joblib.

See also
--------
:class:`mlshell.Workflow` docstring for pipeline prerequisites.

TODO:
Pipeline should contain:
fit():
    .fit()
    .set_params()
    .get_params()
optimize():
    sklearn estimator
validate():
    predict_proba()/predict()
predict():
    .dataset_id
        In `mlshell.workflow` used only in result filenames, can be skipped.
    .pipeline
"""


import os

import joblib
import mlshell
import mlshell.pycnfg as pycnfg
import sklearn.utils.estimator_checks

__all__ = ['Pipeline', 'PipelineProducer']


class Pipeline(object):
    """Unified pipeline interface.

    Implements interface to access arbitrary pipeline.
    Interface: is_classifier, is_regressor, dump and all underlying
        pipeline object methods.

    Attributes
    ----------
    pipeline : object
        Underlying pipeline.
    dataset_id : str
        If pipeline is fitted, train dataset identifier, otherwise None.

    Notes
    -----
    Calling unknown methods are redirected to underlying pipeline object.

    """
    def __init__(self, pipeline=None, dataset_id=None):
        """
        Parameters
        ----------
        pipeline : object with sklearn.pipeline.Pipeline interface, None,
        optional (default=None)
            Pipeline to wrap.
        dataset_id : str, None, optional (default=None),
            Train dataset identifier if pipeline is fitted, otherwise None.

        """
        self.pipeline = pipeline
        self.dataset_id = dataset_id

    def __getattr__(self, name):
        """Redirect unknown methods to pipeline object."""
        def wrapper(*args, **kwargs):
            getattr(self.pipeline, name)(*args, **kwargs)
        return wrapper

    def __hash__(self):
        return str(self.pipeline.get_params())

    def fit(self, *args, **kwargs):
        """Fit pipeline."""
        self.dataset_id = kwargs.pop('dataset_id', None)
        self.pipeline.fit(*args, **kwargs)

    def set_params(self, *args, **kwargs):
        """Set pipeline params."""
        self.dataset_id = None
        self.pipeline.set_params(*args, **kwargs)

    def is_classifier(self):
        """Check if pipeline classifier."""
        return sklearn.base.is_classifier(self.pipeline)

    def is_regressor(self):
        """Check if pipeline regressor."""
        return sklearn.base.is_regressor(self.pipeline)

    def dump(self, filepath, **kwargs):
        """Dump pipeline on disk.

        Parameters
        ----------
        filepath : str
            Filepath without extension.
        **kwargs : dict
        `   Additional kwargs to pass in dump(**kwargs).

        Returns
        -------
        fullpath : str
            Full filepath.

        """
        fullpath = f'{filepath}.model'
        joblib.dump(self.pipeline, fullpath, **kwargs)
        return fullpath


class PipelineProducer(pycnfg.Producer):
    """Class includes methods to produce pipeline.

    Interface: make, load.

    Parameters
    ----------
    objects : dict
        Dictionary with resulted objects from previous executed producers:
        {'section_id__config__id', object,}
    oid : str
        Unique identifier of produced object.
    path_id : str
        Project path identifier in `objects`.
    logger_id : str
        Logger identifier in `objects`.

    Attributes
    ----------
    objects : dict
        Dictionary with resulted objects from previous executed producers:
        {'section_id__config__id', object,}
    oid : str
        Unique identifier of produced object.
    logger : logger object
        Default logger logging.getLogger().
    project_path : str
        Absolute path to project dir.

    """
    _required_parameters = ['objects', 'oid', 'path_id', 'logger_id']

    def __init__(self, objects, oid, path_id, logger_id):
        pycnfg.Producer.__init__(self, objects, oid)
        self.logger = objects[logger_id]
        self.project_path = objects[path_id]

    def make(self, pipeline, steps=None, memory=None, **kwargs):
        """Create pipeline from steps.

        Parameters
        ----------
        pipeline : mlshell.Pipeline
            Pipeline template, will be updated.
        steps: list, class, optional (default=none)
            Steps of pipeline, passed to sklearn.pipeline.Pipeline.
            If class, should support class(**kwargs).steps.
            If None, mlshell.PipelineSteps class is used.
        memory : None, str, joblib.Memory interface, optional (default=None)
            `memory` argument passed to sklearn.pipeline.Pipeline.
            If 'auto', "project_path/temp/pipeline" is used.
        **kwargs : dict
            Additional kwargs to initialize `steps` (if provided as class).

        Returns
        -------
        pipeline : mlshell.Pipeline
            Resulted pipeline.

        Notes
        -----

        """
        self.logger.info("|__  CREATE PIPELINE")
        steps = self._steps_resolve(steps, **kwargs)
        memory = self._memory_resolve(memory)
        pipeline.pipeline = sklearn.pipeline.Pipeline(steps, memory=memory)
        sklearn.utils.estimator_checks.check_estimator(pipeline.pipeline,
                                                       generate_only=False)
        return pipeline

    def load(self, pipeline, filepath, **kwargs):
        """Load fitted model from disk.

        Parameters
        ----------
        pipeline : mlshell.Pipeline
            Pipeline template, will be updated.
        filepath : str
            Absolute path load file or relative to 'self.project_dir' started
            with './'.
        kwargs : dict
            Additional parameters to pass in load().

        Returns
        -------
        pipeline : mlshell.Pipeline
            Resulted pipeline.

        """
        self.logger.info("|__  LOAD PIPELINE")
        if filepath.startswith('./'):
            filepath = f"{self.project_path}/{filepath[2:]}"

        pipeline.pipeline = joblib.load(filepath, **kwargs)
        self.logger.info('Load fitted model from file:\n'
                         '    {}'.format(filepath))
        return pipeline

    def info(self, pipeline, **kwargs):
        """Log pipeline info.

        Parameters
        ----------
        pipeline : mlshell.Pipeline
            Pipeline to explore.
        **kwargs : kwargs
            Additional parameters to pass in low-level functions.

        Returns
        -------
        pipeline : mlshell.Pipeline
            For compliance with producer logic.

        """
        self._print_steps(pipeline.pipeline, **kwargs)
        return pipeline

    def _memory_resolve(self, memory):
        if memory == 'auto':
            memory = f"{self.project_path}/temp/pipeline"
        if not os.path.exists(memory):
            # Create temp dir for cache if not exist.
            os.makedirs(memory)
        return memory

    def _steps_resolve(self, steps, **kwargs):
        """Prepare pipeline steps.

        Returns
        -------
        steps: list
            sklearn.pipeline.Pipeline steps.

        """
        if isinstance(steps, list):
            steps = steps
        else:
            if steps is None:
                clss = mlshell.PipelineSteps
            else:
                clss = steps
            steps = clss(**kwargs).steps
        return steps

    def _print_steps(self, pipeline, **kwargs):
        """"Nice print of pipeline steps."""
        params = pipeline.pipeline.get_params()
        self.logger.debug('Pipeline steps:')
        if 'steps' not in params:
            return
        for i, step in enumerate(params['steps']):
            step_name = step[0]
            # step_hp = {key: params[key] for key in params.keys()
            #            if step_name + '__' in key}
            self.logger.debug(f"  ({i})  {step[0]}\n"
                              f"    {step[1]}")
            self.logger.debug('    hp:\n'
                              '   {jsbeautifier.beautify(str(step_hp))}')
        self.logger.debug('+' * 100)
        return


if __name__ == '__main__':
    pass
