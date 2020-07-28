"""
The :mod:`mlshell.producers.workflow` contains examples of `Workflow` class to
produce results for typical machine learning task.

:class:`mlshell.Workflow` class uses unified interface to work with underlying
pipelines/datasets/metrics. Current implementation specifies methods to
fit/predict/optimize/validate/dump pipeline and plot results.

Workflow support multi-stage optimization for a pipeline-dataset pair. Each
stage combined results with previous to find the best hp combination and apply
it in the next stage.

"""


import inspect
import os
import pathlib
import threading

import mlshell
import numpy as np
import pandas as pd
import pycnfg
import sklearn

__all__ = ['Workflow']


class Workflow(pycnfg.Producer):
    """Interface to ML task.

    Interface: fit, predict, optimize, validate, dump, plot.

    Parameters
    ----------
    objects : dict
        Dictionary with resulted objects from previous executed producers:
        {'section_id__config__id', object}.
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
    logger : :class:`logging.Logger`
        Logger.
    project_path : str
        Absolute path to project dir.

    See also
    --------
    :class:`mlshell.Dataset` : Dataset interface.
    :class:`mlshell.Metric` : Metric inteface.
    :class:`mlshell.Pipeline` : Pipeline inteface.

    """
    _required_parameters = ['objects', 'oid', 'path_id', 'logger_id']

    def __init__(self, objects, oid, path_id, logger_id):
        pycnfg.Producer.__init__(self, objects, oid)
        self.logger = objects[logger_id]
        self.project_path = objects[path_id]
        self._optional()

    def _optional(self):
        # Turn on: inf as NaN.
        pd.options.mode.use_inf_as_na = True
        # Handle numpy errors.
        np.seterr(all='call')
        self._check_results_size(self.project_path)
        self._np_error_stat = {}
        np.seterrcall(self._np_error_callback)

    def _check_results_size(self, project_path):
        root_directory = pathlib.Path(f"{project_path}/results")
        size = sum(f.stat().st_size for f in root_directory.glob('**/*')
                   if f.is_file())
        size_mb = size/(2**30)
        n = 5  # Check if dir > n Mb.
        if size_mb > n:
            self.logger.warning(f"Warning: results/ directory size "
                                f"{size_mb:.2f}Gb more than {n}Gb")

    def _np_error_callback(self, *args):
        """Numpy errors handler, count errors by type."""
        if args[0] in self._np_error_stat.keys():
            self._np_error_stat[args[0]] += 1
        else:
            self._np_error_stat[args[0]] = 1

    def fit(self, res, pipeline_id, dataset_id, subset_id='train',
            hp=None, resolver=None, resolve_params=None,
            fit_params=None):
        """Fit pipeline.

        Parameters
        ----------
        res : dict
            For compliance with producer logic.
        pipeline_id : str
            Pipeline identifier in ``objects``. Will be fitted on `dataset_id__
            subset_id`: ``pipeline.fit(subset.x, subset.y, **fit_params)`` .
        dataset_id : str
            Dataset identifier in ``objects``.
        subset_id : str, optional (default='train')
            Data subset identifier to fit on. If '', use full dataset.
        hp : dict, optional (default=None)
            Hyper-parameters to use in pipeline: {`hp_name`: val/container}.
            If container provided, zero position will be used. If None, {}
        resolver : :class:`mlshell.model_selection.Resolver`, optional
                (default=None)
            If hp value = 'auto', hp will be resolved: ``resolver.resolve()``.
            Auto initialized if necessary. :class:`mlshell.model_selection.
            Resolver` if None.
        resolve_params : dict, optional (default=None)
            Additional kwargs to pass in: ``resolver.resolve(*args,
            **resolve_params[hp_name])``. If None, {}.
        fit_params : dict, optional (default=None)
            Additional kwargs to pass in ``pipeline.fit(*args, **fit_params)``.
            If None, {}.

        Returns
        -------
        res : dict
            Unchanged input, for compliance with producer logic.

        Notes
        -----
        Pipeline updated in ``objects`` attribute.

        See Also
        --------
        :class:`mlshell.model_selection.Resolver` : Hp resolver.

        """
        self.logger.info("|__ FIT PIPELINE")
        if hp is None:
            hp = {}
        if resolver is None:
            resolver = mlshell.model_selection.Resolver
        if inspect.isclass(resolver):
            resolver = resolver()
        if resolve_params is None:
            resolve_params = {}
        if fit_params is None:
            fit_params = {}

        pipeline = self.objects[pipeline_id]
        dataset = self.objects[dataset_id]
        pipeline = self._set_hp(
            hp, pipeline, resolver, dataset, resolve_params)

        train = dataset.subset(subset_id)
        pipeline.fit(train.x, train.y, **fit_params)
        pipeline.dataset_id = train.oid
        self.objects[pipeline_id] = pipeline
        return res

    def optimize(self, res, pipeline_id, dataset_id, subset_id='train',
                 hp_grid=None, scoring=None, resolver=None,
                 resolve_params=None, optimizer=None, gs_params=None,
                 fit_params=None, dirpath=None, dump_params=None):
        """Optimize pipeline.

        Parameters
        ----------
        res : dict
            For compliance with producer logic.
        pipeline_id : str
            Pipeline identifier in ``objects``. Will be cross-validate on
            `dataset_id__subset_id`: ``optimizer.fit(subset.x, subset.y,
            **fit_params)`` .
        dataset_id : str
            Dataset identifier in `objects`.
        subset_id : str, optional (default='train')
            Data subset identifier to CV on. If '', use full dataset.
        hp_grid : dict, optional (default=None)
            Hyper-parameters to grid search: {`hp_name`: optimizer format}.
            If None, {}.
        scoring : List of str, optional (default=None)
            List of 'metric_id' to use in optimizer. Known 'metric_id' will be
            resolved via `objects` or sklearn built-in, otherwise raise
            ``KeyError``. If None, 'accuracy' or 'r2' depends on pipeline
            estimator type.
        resolver : :class:`mlshell.model_selection.Resolver`, optional
                (default=None)
            If hp value = ['auto'] in ``hp_grid``, hp will be resolved via
            ``resolver.resolve()``. Auto initialized if class provided.
            If None, :class:`mlshell.model_selection.Resolver` used.
        resolve_params : dict, optional (default=None)
            Additional kwargs to pass in ``resolver.resolve(*args,
            **resolve_params[hp_name])`` . If None, {}.
        optimizer : :class:`mlshell.model_selection.Optimizer``, optional
                (default=None)
            Class to optimize ``hp_grid``. Will be called ``optimizer(pipeline,
            hp_grid, scoring, **gs_params).fit(x, y, **fit_params)``. If None,
            :class:`mlshell.model_selection.RandomizedSearchOptimizer` .
        fit_params : dict, optional (default=None)
            Additional kwargs to pass in ``optimizer.fit(*args,
            **fit_params)``. If None, {}.
        gs_params : dict, optional (default=None)
            Additional kwargs to ``optimizer(pipeline, hp_grid, scoring,
            **gs_params)`` initialization. If None, {}.
        dirpath : str, optional (default=None)
            Absolute path to the dump result 'runs' dir or relative to
            'project__path' started with './'. If None, "project__path
            /results/runs" is used. See Notes for runs description.
        dump_params: dict, optional (default=None)
            Additional kwargs to pass in ``optimizer.dump_runs(**dump_params)``.
            If None, {}.

        Returns
        -------
        res : dict
            Input`s key added/updated: {

            'runs': dict
                Storage of optimization results for pipeline-data pair.
                {'pipeline_id|dataset_id|subset_id':
                    optimizer.update_best output}

            }

        Notes
        -----
        Optimization flow:

        * Call grid search.
         ``optimizer(pipeline, hp_grid, scoring, **gs_params)
         .fit(x, y, **fit_params)`` .
        * Call dump runs.
         ``optimizer.dump_runs(logger, dirpath, **dump_params)``,
         where each run = probing one hp combination.
        * Combine optimization results with previous for pipeline-data pair:
         ``optimizer.update_best(prev_runs)`` .
        * Upfate pipeline object in ``objects``.
         Onle if 'best_estimator_' in 'runs'.

        See Also
        --------
        :class:`mlshell.model_selection.Resolver` : Hp resolver.
        :class:`mlshell.model_selection.Optimizer` : Hp optimizer.


        """
        self.logger.info("|__ OPTIMIZE HYPER-PARAMETERS")
        if hp_grid is None:
            hp_grid = {}
        if resolver is None:
            resolver = mlshell.model_selection.Resolver
        if inspect.isclass(resolver):
            resolver = resolver()
        if resolve_params is None:
            resolve_params = {}
        if fit_params is None:
            fit_params = {}
        if optimizer is None:
            optimizer = mlshell.model_selection.RandomizedSearchOptimizer
        if dirpath is None:
            dirpath = f"{self.project_path}/results/runs"
        elif dirpath.startswith('./'):
            dirpath = f"{self.project_path}/{dirpath[2:]}"
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        pipeline = self.objects[pipeline_id]
        dataset = self.objects[dataset_id]
        # Resolve and set hp. Otherwise could be problem if hp_grid={}, as
        # pipeline initially could have unresolved.
        pipeline = self._set_hp(
            {}, pipeline, resolver, dataset, resolve_params)
        # Resolve hp_grid.
        hp_grid = self._resolve_hp(
            hp_grid, pipeline, resolver, dataset, resolve_params)
        # Resolve scoring.
        scoring = self._resolve_scoring(scoring, pipeline)

        train = dataset.subset(subset_id)
        optimizer = optimizer(pipeline, hp_grid, scoring, **gs_params)
        optimizer.fit(train.x, train.y, **fit_params)

        self.logger.info("    |__ DUMP RUNS")
        optimizer.dump_runs(self.logger, dirpath, pipeline, dataset,
                            **dump_params)

        if 'runs' not in res:
            res['runs'] = {}
        runs = res['runs']
        key = f"{pipeline_id}|{dataset_id}|{subset_id}"
        runs[key] = optimizer.update_best(runs.get(key, {}))
        if 'best_estimator_' in runs[key]:
            self.objects[pipeline_id] = runs[key].get('best_estimator_')
            self.objects[pipeline_id].dataset_id = train.oid
        return res

    def validate(self, res, pipeline_id, dataset_id, metric_id,
                 subset_id=('train', 'test'), validator=None):
        """Make and score prediction.

        Parameters
        ----------
        res : dict
            For compliance with producer logic.
        pipeline_id : str
            Pipeline identifier in ``objects``. Will be validated on
            `dataset_id__subset_id`.
        dataset_id : str
            Dataset identifier in `objects`.
        subset_id : str, tuple of str, optional (default=('train', 'test'))
            Data subset(s) identifier(s) to validate on. '' for full dataset.
        metric_id : srt, list of str
            Metric(s) identifier in `objects`.
        validator : :class:`mlshell.model_selection.Validator`, optional
        (default=None)
            Auto initialized if class provided. If None,
            :class:`mlshell.model_selection.Validator` .

        Returns
        -------
        res : dict
            Unchanged input, for compliance with producer logic.

        """
        self.logger.info("|__ VALIDATE")
        if validator is None:
            validator = mlshell.model_selection.Validator
        if inspect.isclass(validator):
            validator = validator()
        if not isinstance(metric_id, list):
            metric_id = [metric_id]
        if not isinstance(subset_id, list):
            subset_id = [subset_id]

        dataset = self.objects[dataset_id]
        pipeline = self.objects[pipeline_id]
        metrics = [self.objects[i] for i in metric_id]
        subsets = [dataset.subset(id_) for id_ in subset_id]

        validator.validate(pipeline, metrics, subsets,
                           self.logger)
        return res

    def dump(self, res, pipeline_id, dirpath=None, **kwargs):
        """Dump pipeline.

        Parameters
        ----------
        res : dict
            For compliance with producer logic.
        pipeline_id : str
            Pipeline identifier in ``objects``. Will be dumped via
            ``pipeline.dump(**kwargs)`` .
        dirpath : str, optional(default=None)
            Absolute path to dump dir or relative to 'project__path' started
            with './'. If None,"project__path/results/models" is used.
        **kwargs: dict
            Additional kwargs to pass in ``pipeline.dump(**kwargs)`` .

        Returns
        -------
        res : dict
            Unchanged input, for compliance with producer logic.

        Notes
        -----
        Resulted filename includes prefix:
        ``{workflow_id}_{pipeline_id}_{fit_dataset_id}_{best_score}_
        {pipeline_hash}_{fit_dataset_hash}``.

        If pipeline not fitted, 'fit_dataset_id' = None .
        The 'best_score' available after optimize step(s) only if optimizer
        supported.

        """
        self.logger.info("|__ DUMP MODEL")
        if dirpath is None:
            dirpath = f"{self.project_path}/results/models"
        elif dirpath.startswith('./'):
            dirpath = f"{self.project_path}/{dirpath[2:]}"
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        pipeline = self.objects[pipeline_id]
        filepath = self._prefix(res, dirpath, pipeline, pipeline_id)
        fullpath = pipeline.dump(filepath, **kwargs)
        self.logger.log(25, f"Save fitted model to file:\n"
                            f"    {fullpath}")
        return res

    def predict(self, res, pipeline_id, dataset_id, subset_id='test',
                dirpath=None, **kwargs):
        """Make and dump prediction.

        Parameters
        ----------
        res : dict
            For compliance with producer logic.
        pipeline_id : str
            Pipeline identifier in `objects` to make prediction on
            `dataset_id__subset_id`: ``pipeline.predict(subset.x)`` .
        dataset_id : str
            Dataset identifier in `objects`.
        subset_id : str, optional (default='test')
            Data subset identifier to predict on. If '', use full dataset.
        dirpath : str, optional (default=None)
            Absolute path to dump dir or relative to 'project__path' started
            with './'. If None, "project__path/results/models" is used.
        **kwargs: dict
            Additional kwargs to pass in ``dataset.dump_pred(**kwargs)`` .

        Returns
        -------
        res : dict
            Unchanged input, for compliance with producer logic.

        Notes
        -----
        Resulted filename includes prefix:
        ``{workflow_id}_{pipeline_id}_{fit_dataset_id}_{best_score}
        _{pipeline_hash}_{fit_dataset_hash}
        _{predict_dataset_id}_{predict_dataset_hash}``

        If pipeline not fitted, 'fit_dataset_id' = None .
        The `best_score` available only after optimize step(s) if optimizer
        supported.

        """
        self.logger.info("|__ PREDICT")
        if dirpath is None:
            dirpath = f"{self.project_path}/results/models"
        elif dirpath.startswith('./'):
            dirpath = f"{self.project_path}/{dirpath[2:]}"
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        pipeline = self.objects[pipeline_id]
        dataset = self.objects[dataset_id]
        test = dataset.subset(subset_id)
        y_pred = pipeline.predict(test.x)

        filepath = self._prefix(res, dirpath, pipeline, pipeline_id,
                                test, test.oid)
        fullpath = test.dump_pred(filepath, y_pred, **kwargs)
        self.logger.log(25, f"Save predictions for {dataset_id} to file:"
                            f"\n    {fullpath}")
        return res

    def plot(self, res, pipeline_id, dataset_id, metric_id, validator=None,
             subset_id=('train', 'test'), plotter=None, **kwargs):
        """Plot metrics.

        Parameters
        ----------
        res : dict
            For compliance with producer logic.
        pipeline_id : str
            Pipeline identifier in ``objects``.
        dataset_id : str
            Dataset identifier in ``objects``.
        subset_id : str, tuple of str, optional (default=('train', 'test'))
            Data subset(s) identifier(s) to plot on. Set '' for full dataset.
        metric_id : srt, list of str
            Metric(s) identifier in `objects`.
        validator : :class:`mlshell.model_selection.Validator`, optional
        (default=None)
            Auto initialized if class provided. If None,
            :class:`mlshell.model_selection.Validator` .
        plotter : :class:`mlshell.plot.Plotter`, optional (default=None)
            Auto initialized if class provided. If None,
            :class:`mlshell.plot.Plotter` .
        **kwargs: dict
            Additional kwargs to pass in ``plotter.plot(**kwargs)`` .

        Returns
        -------
        res : dict
            Unchanged input, for compliance with producer logic.

        See Also
        --------
        :class:`mlshell.plot.Plotter` : Metric plotter.

        """
        self.logger.info("|__ PLOT")
        if validator is None:
            validator = mlshell.model_selection.Validator
        if inspect.isclass(validator):
            validator = validator()
        if plotter is None:
            plotter = mlshell.plot.Plotter
        if inspect.isclass(plotter):
            plotter = plotter()
        if not isinstance(metric_id, list):
            metric_id = [metric_id]
        if not isinstance(subset_id, list):
            subset_id = [subset_id]

        dataset = self.objects[dataset_id]
        pipeline = self.objects[pipeline_id]
        metrics = [self.objects[i] for i in metric_id]
        subsets = {id_: dataset.subset(id_) for id_ in subset_id}
        runs = res.get('runs', {})
        subruns = {id_: runs.get(f"{pipeline_id}|{dataset_id}|{id_}", {})
                   for id_ in subset_id}

        args = (subruns, pipeline, metrics, subsets, validator, self.logger)
        threading.Thread(target=plotter.plot, args=args, kwargs=kwargs,
                         daemon=True).start()
        return res

    # ========================== fit/optimize =================================
    def _set_hp(self, hp, pipeline, resolver, dataset, resolve_params):
        """Get => update => resolve => set pipeline hp."""
        _hp_full = pipeline.get_params()
        _hp_full.update(self._get_zero_position(hp))
        hp = self._resolve_hp(_hp_full, pipeline, resolver, dataset,
                              **resolve_params)
        pipeline.set_params(**hp)
        return pipeline

    def _get_zero_position(self, hp):
        """Get zero position if hp_grid provided.

        Notes
        -----
        In case of generator/iterator in hp value,  hp_grid changes will be
        irreversible.

        """
        # Get zero position params from hp.
        zero_hp = {}
        for name, vals in hp.items():
            # Check if not distribution in hp.
            if hasattr(type(vals), '__iter__'):
                # Container type.
                iterator = iter(vals)
                zero_hp.update(**{name: iterator.__next__()})
        return zero_hp

    def _resolve_hp(self, hp, pipeline, resolver, dataset, resolve_params):
        """Resolve hyper-parameter based on dataset value.

        For example, categorical features indices are dataset dependent.
        Resolve lets to set it before fit/optimize step.

        Parameters
        ----------
        hp : dict
            {hp_name: val/container}. If val=='auto'/['auto'] hp will be
            resolved.
        pipeline : :class:`mlshell.Pipeline`
            Pipeline, passed to `resolver`
        resolver : :class:`mlshell.model_selection.Resolver`
            Interface to resolve hp.
        dataset : :class:`mlshell.Dataset`
            Dataset, passed to `resolver`.
        **resolve_params: : dict
            Additional parameters to pass in ``resolver.resolve(*arg,
            **resolve_params['hp_name'])`` for specific hp: {hp_name: kwargs}.

        Returns
        -------
        hp: dict
            Resolved input hyper-parameters.

        """
        for hp_name, val in hp.items():
            if val == 'auto' or val == ['auto']:
                kwargs = resolve_params.get(hp_name, {})
                value = resolver.resolve(hp_name, val, pipeline, dataset,
                                         **kwargs)
                hp[hp_name] = [value] if val == ['auto'] else value

        return hp

    def _resolve_scoring(self, scoring, pipeline):
        """Resolve scoring for grid search.

        Notes
        -----
        If None, 'accuracy' or 'r2' depends on estimator type.
        If list, resolve known metric id via `objects` and sklearn built-in,
        otherwise passed unchanged.

        """
        if scoring is None:
            # Hard-code (default estimator could not exist).
            if pipeline.is_classifier():
                scoring = 'accuracy'
            elif pipeline.is_regressor():
                scoring = 'r2'
        elif isinstance(scoring, list):
            # Resolve if exist, else use sklearn built-in.
            scoring = {i: self.objects[i] if i in self.objects
                       else sklearn.metrics.SCORERS[i]
                       for i in scoring}
        return scoring

    # ========================== dump/predict =================================
    def _prefix(self, res, dirpath, pipeline, pipeline_id,
                pred_dataset=0, pred_dataset_id=''):
        """Generate informative file prefix."""
        pred_dataset_hash = hash(pred_dataset)
        fit_dataset_id = getattr(pipeline, 'dataset_id', None)
        fit_dataset_hash = hash(self.objects.get(fit_dataset_id, 0))
        best_score = str(res.get('runs', {})
                            .get((pipeline_id, fit_dataset_id), {})
                            .get('best_score_', '')
                         ).lower()
        filepath = f"{dirpath}/{self.oid}_{pipeline_id}_{fit_dataset_id}_" \
                   f"{best_score}_{hash(pipeline)}_{fit_dataset_hash}_" \
                   f"{pred_dataset_id}_{pred_dataset_hash}"
        return filepath


if __name__ == '__main__':
    pass
