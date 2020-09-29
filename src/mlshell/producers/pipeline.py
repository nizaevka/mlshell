"""
The :mod:`mlshell.producers.pipeline` contains examples of `Pipeline` class to
create empty pipeline object and `PipelineProducer` class to fill it.

:class:`mlshell.Pipeline` proposes unified interface to work with
underlying pipeline. Intended to be used in :mod:`mlshell.Workflow`.
For new pipeline formats no need to edit `Workflow` class, just adapt in
compliance to `Pipeline` interface.

:class:`mlshell.PipelineProducer` specifies methods to create/load
pipeline. Model loading currently implemented via :mod:`joblib` and model
creation via :class:`sklearn.pipeline.Pipeline` .

"""


import os

import joblib
import mlshell
import pycnfg
import sklearn.utils.estimator_checks

__all__ = ['Pipeline', 'PipelineProducer']


class Pipeline(object):
    """Unified pipeline interface.

    Implements interface to access arbitrary pipeline.
    Interface: is_classifier, is_regressor, dump, set_params and all underlying
    pipeline object methods.

    Attributes
    ----------
    pipeline : :mod:`sklearn` estimator
        Underlying pipeline.
    dataset_id : str
        If pipeline is fitted, train dataset identifier, otherwise None.

    Notes
    -----
    Calling unspecified methods are redirected to underlying pipeline object.

    """
    def __init__(self, pipeline=None, oid=None, dataset_id=None):
        """
        Parameters
        ----------
        pipeline : :mod:`sklearn` estimator, optional (default=None)
            Pipeline to wrap.
        oid : str
            Instance identifier.
        dataset_id : str, optional (default=None),
            Train dataset identifier if pipeline is fitted, otherwise None.

        """
        self.pipeline = pipeline
        self.oid = oid
        self.dataset_id = dataset_id

    def __hash__(self):
        s = str(self.pipeline.get_params())
        return hash(s)

    def __getattr__(self, name):
        """Redirect unknown methods to pipeline object."""
        def wrapper(*args, **kwargs):
            return getattr(self.pipeline, name)(*args, **kwargs)
        return wrapper

    def __getstate__(self):
        # Allow pickle.
        return self.__dict__

    def __setstate__(self, d):
        # Allow unpickle.
        self.__dict__ = d

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
        """Dump the pipeline on disk.

        Parameters
        ----------
        filepath : str
            File path without extension.
        **kwargs : dict
        `   Additional kwargs to pass in dump(**kwargs).

        Returns
        -------
        fullpath : str
            Full file path.

        """
        fullpath = f'{filepath}.model'
        joblib.dump(self.pipeline, fullpath, **kwargs)
        return fullpath


class PipelineProducer(pycnfg.Producer):
    """Factory to produce pipeline.

    Interface: set, make, load, info.

    Parameters
    ----------
    objects : dict
        Dictionary with objects from previous executed producers:
        {'section_id__config__id', object,}
    oid : str
        Unique identifier of produced object.
    path_id : str, optional (default='default')
        Project path identifier in `objects`.
    logger_id : str, optional (default='default')
        Logger identifier in `objects`.

    Attributes
    ----------
    objects : dict
        Dictionary with objects from previous executed producers:
        {'section_id__config__id', object,}
    oid : str
        Unique identifier of produced object.
    logger : :class:`logging.Logger`
        Logger.
    project_path : str
        Absolute path to project dir.

    """
    _required_parameters = ['objects', 'oid', 'path_id', 'logger_id']

    def __init__(self, objects, oid, path_id='path__default',
                 logger_id='logger__default'):
        pycnfg.Producer.__init__(self, objects, oid, path_id=path_id,
                                 logger_id=logger_id)

    def set(self, pipeline, estimator):
        """Set estimator as pipeline.

        Parameters
        ----------
        pipeline : :class:`mlshell.Pipeline`
            Pipeline object, will be updated.
        estimator : :mod:`sklearn` estimator
            Estimator to set in :class:`mlshell.Pipeline` interface.

        Returns
        -------
        pipeline : :class:`mlshell.Pipeline`
            Resulted pipeline.

        """
        pipeline.pipeline = estimator
        return pipeline

    def make(self, pipeline, steps=None, estimator=None, memory=None, **kwargs):
        """Create pipeline from steps.

        Parameters
        ----------
        pipeline : :class:`mlshell.Pipeline`
            Pipeline object, will be updated.
        steps: list, class, optional (default=None)
            Pipeline steps to pass in :class:`sklearn.pipeline.Pipeline` .
            Could be a class with ``steps`` attribute, will be initialized
            ``steps(estimator, **kwargs)``. If None, :class:`mlshell.pipeline.
            Steps` is used. Set ``[]`` to use``estimator`` direct as pipeline.
        estimator : :mod:`sklearn` estimator
            Estimator to use in the last step if ``steps`` is a class or direct
            if ``steps=[]``.
        memory : str, :class:`joblib.Memory` interface, optional (default=None)
            `memory` argument passed to :class:`sklearn.pipeline.Pipeline` .
            If 'auto', "project_path/.temp/pipeline" is used.
        **kwargs : dict
            Additional kwargs to initialize `steps` (if provided as class).

        Returns
        -------
        pipeline : :class:`mlshell.Pipeline`
            Resulted pipeline.

        """
        steps = self._steps_resolve(steps, estimator=estimator, **kwargs)
        if steps:
            memory = self._memory_resolve(memory)
            pipeline.pipeline = sklearn.pipeline.Pipeline(steps, memory=memory)
        else:
            pipeline.pipeline = estimator
        return pipeline

    def load(self, pipeline, filepath, **kwargs):
        """Load fitted model from disk.

        Parameters
        ----------
        pipeline : :class:`mlshell.Pipeline`
            Pipeline object, will be updated.
        filepath : str
            Absolute path to load file or relative to 'project__path'
            started with './'.
        kwargs : dict
            Additional parameters to pass in load().

        Returns
        -------
        pipeline : :class:`mlshell.Pipeline`
            Resulted pipeline.

        """
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
        pipeline : :class:`mlshell.Pipeline`
            Pipeline to explore (if 'steps' attribute available).
        **kwargs : dict
            Additional parameters to pass in low-level functions.

        Returns
        -------
        pipeline : :class:`mlshell.Pipeline`
            For compliance with producer logic.

        """
        self._print_steps(pipeline, **kwargs)
        return pipeline

    # ================================ make ===================================
    def _steps_resolve(self, steps, **kwargs):
        """Prepare pipeline steps.

        Returns
        -------
        steps: list
            :class:`sklearn.pipeline.Pipeline` steps.

        """
        if isinstance(steps, list):
            steps = steps
        else:
            if steps is None:
                clss = mlshell.pipeline.Steps
            else:
                clss = steps
            steps = clss(**kwargs).steps
        return steps

    def _memory_resolve(self, memory):
        if memory == 'auto':
            memory = f"{self.project_path}/.temp/pipeline"
        if memory is not None and not os.path.exists(memory):
            # Create temp dir for cache if not exist.
            os.makedirs(memory)
        return memory

    # ================================ info ===================================
    def _print_steps(self, pipeline, **kwargs):
        """"Nice print of pipeline steps."""
        params = pipeline.get_params()
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
