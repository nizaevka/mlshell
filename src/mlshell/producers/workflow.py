"""
The :mod:`mlshell.workflow` contains examples of `Workflow` class to produce
results for typical machine learning task .

`Workflow` class uses unified interface to work with underlying
pipelines/datasets/metrics. Current implementation specifies methods to
fit/predict/optimize/validate/dump pipeline and plot results.

Workflow support multi-stage optimization for some pipeline-dataset
pair. Each stage results combined with previous stage to find the best
hp combination and use it in the next stage.

# TODO: Move out to related methods.
(dict): if user skip declaration for any parameter the default one will be used.

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


import mlshell.pycnfg as pycnfg
import mlshell
import hashlib
import pathlib
import inspect
import jsbeautifier
import numpy as np
import pandas as pd
import os
import threading
import time
import sklearn
import copy

__all__ = ['Workflow']


def checker(func, options=None):
    """Decorator.

    Logs:
    * Alteration in objects.
    * Numpy errors.

     """
    if options is None:
        options = []
    def wrapper(*args, **kwargs):
        self = args[0]
        hash_before = {key: hash(val) for key, val in self.object.items()}
        func(*args, **kwargs)
        hash_after = {key: hash(val) for key, val in self.object.items()}
        hash_diff = {key: {'before': hash_before[key],
                           'after': hash_after[key]} for key in hash_before
                            if hash_before[key]!=hash_after[key] }
        if hash_diff:
            self.logger.info(f"Object(s) hash changed:\n"
                             f"    {hash_diff}")
        if self._np_error_stat:
            self.logger.info('Numpy error(s) occurs:\n'
                             '    {}'.format(self._np_error_stat))
            self._np_error_stat = {}
    return wrapper


class Workflow(pycnfg.Producer):
    """Interface to produce ml results.

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
    logger : logger object
        Default logger logging.getLogger().
    project_path : str
        Absolute path to project dir.

    See also
    --------
    :class:`mlshell.Dataset` dataset interface.
    :class:`mlshell.Pipeline` pipeline inteface.
    :class:`mlshell.Optimizer` optimizer inteface.
    :class:`mlshell.Resolver` default hp resolver.

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
        """Numpy errors handler, count errors by type"""
        if args[0] in self._np_error_stat.keys():
            self._np_error_stat[args[0]] += 1
        else:
            self._np_error_stat[args[0]] = 1

    @checker
    # @memory_profiler
    def fit(self, res, pipeline_id, dataset_id, subset_id='train',
            hp=None, resolver=None, resolve_params=None,
            fit_params=None):
        """Fit pipeline.

        Parameters
        ----------
        res : dict
            For compliance with producer logic.
        pipeline_id : str
            Pipeline identifier in `objects`. Will be fitted on
            `dataset_id__subset_id`.
        dataset_id : str
            Dataset identifier in `objects`.
        subset_id : str, optional (default='train')
            Data subset identifier to fit on. If '', use full dataset.
        hp : dict, None, optional (default=None)
            Hyper-parameters to use in pipeline: {`hp_name`: val/container}.
            If container provided, zero position will be used. If None, {}
        resolver : mlshell.Resolver, None, optional (default=None)
            If hp value = 'auto', hp will be resolved via `resolver`.resolve().
            Auto initialized if necessary. If None, mlshell.Resolver used.
        resolve_params : dict, None, optional (default=None)
            Additional kwargs to pass in `resolver`.resolve(*args,
            **resolve_params[hp_name]). If None, {}.
        fit_params : dict, None, optional (default=None)
            Additional kwargs to pass in `pipeline`.fit(*args,
            **fit_params). If None, {}.

        Returns
        -------
        res : dict
            Unchanged input, for compliance with producer logic.

        Notes
        -----
        Pipeline updated in `objects` attribute.

        """
        self.logger.info("|__ FIT PIPELINE")
        if hp is None:
            hp = {}
        if resolver is None:
            resolver = mlshell.Resolver
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
        pipeline.dataset_id = dataset_id
        self.objects[pipeline_id] = pipeline
        return res

    @checker
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
            Pipeline identifier in `objects`. Will be cross-validate on
            `dataset_id__subset_id`.
        dataset_id : str
            Dataset identifier in `objects`.
        subset_id : str, optional (default='train')
            Data subset identifier to CV on. If '', use full dataset.
        hp_grid : dict, None, optional (default=None)
            Hyper-parameters to grid search: {`hp_name`: optimizer format}.
            If None, {}.
        scoring : List of str, None, optimizer format, optional (default=None)
            If None, 'accuracy' or 'r2' depends on estimator type. If list of
            str, try to resolve via `objects`/sklearn built-in: {'metric_id':
            resolved scorer}. Otherwise passed to optimizer unchanged.
        resolver : mlshell.Resolver, None, optional (default=None)
            If hp value = ['auto'] in `hp_grid`, hp will be resolved via
            `resolver`.resolve(). Auto initialized if class provided. If None,
            mlshell.Resolver used.
        resolve_params : dict, None, optional (default=None)
            Additional kwargs to pass in `resolver`.resolve(*args,
            **resolve_params[hp_name]). If None, {}.
        optimizer : mlshell.Optimizer, None, optional (default=None)
            Class to optimize `hp_grid`. If None, mlshell.Optimizer.
            optimizer(pipeline, hp_grid, scoring, **gs_params).fit(x, y,
            **fit_params) will be called.
        fit_params : dict, None, optional (default=None)
            Additional kwargs to pass in `optimizer`.fit(*args,
            **fit_params). If None, {}.
        gs_params :  dict, None, optional (default=None)
            Additional kwargs to `optimizer`(pipeline, hp_grid, scoring,
             **gs_params) initialization. If None, {}.
        dirpath : str, optional (default=None)
            Absolute path to the dump result 'runs' dir or relative to
            'self.project_dir' started with './'. If None, "self.project_path
            /results/runs" is used. See Notes for runs description.
        dump_params: dict, optional (default=None)
        `   Additional kwargs to pass in optimizer.dump_runs(**dump_params). If
            None, {}.

        Returns
        -------
        res : dict
            Input`s key added/updated:
            {'runs': {'pipeline_id|dataset_id': optimizer.update_best output}}.

        Notes
        -----
        Optimize step:
        * Call grid search: optimizer(pipeline, hp_grid, scoring, **gs_params)
        .fit(x, y, **fit_params).
        * Call dump runs: optimizer.dump_runs(self.logger, dirpath,
         **dump_params), where run = probing hp combination.
        * Call optimizer.update_best(prev_runs) to combine previous
        optimization stage for specific dataset-pipeline pair and find best.
        * Pipeline object updated in `self.objects` only if 'runs' contain
        'best_estimator_'.

        See Also
        --------
        :class:`mlshell.Optimizer` optimizer interface.

        """
        self.logger.info("|__ OPTIMIZE HYPER-PARAMETERS")
        if hp_grid is None:
            hp_grid = {}
        if resolver is None:
            resolver = mlshell.Resolver
        if inspect.isclass(resolver):
            resolver = resolver()
        if resolve_params is None:
            resolve_params = {}
        if fit_params is None:
            fit_params = {}
        if optimizer is None:
            optimizer = mlshell.RandomizedSearchOptimizer
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
        key = f"{pipeline_id}|{dataset_id}"
        runs[key] = optimizer.update_best(runs.get(key, {}))
        if 'best_estimator_' in runs[key]:
            self.objects[pipeline_id] = runs[key].get('best_estimator_')
        return res

    # @memory_profiler
    def validate(self, res, pipeline_id, dataset_id, metric_id,
                 subset_id=('train', 'test'), validator=None):
        """Predict and score on validation set.

        Parameters
        ----------
        res : dict
            For compliance with producer logic.
        pipeline_id : str
            Pipeline identifier in `objects`. Will be validated on
            `dataset_id__subset_id`.
        dataset_id : str
            Dataset identifier in `objects`.
        subset_id : str, tuple of str, optional (default=('train', 'test'))
            Data subset(s) identifier(s) to validate on. '' for full dataset.
        metric_id : srt, list of str
            Metric(s) identifier in `objects`.
        validator : mlshell.Validator, None, optional (default=None)
            Auto initialized if class provided. If None, mlshell.Validator.

        Returns
        -------
        res : dict
            Unchanged input, for compliance with producer logic.

        """
        self.logger.info("|__ VALIDATE")
        if validator is None:
            validator = mlshell.Validator
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
        """Dump fitted model.

        Parameters
        ----------
        res : dict
            For compliance with producer logic.
        pipeline_id : str
            Pipeline identifier in `objects`. Will be dumped via pipeline.dump.
        dirpath : str, optional(default=None)
            Absolute path dump dir or relative to 'self.project_dir' started
            with './'. If None,"self.project_path/results/models" is used.
        **kwargs: dict
        `   Additional kwargs to pass in pipeline.dump(**kwargs).

        Returns
        -------
        res : dict
            Unchanged input, for compliance with producer logic.

        Notes
        -----
        Resulted filename includes prefix:
        "{workflow_id}_{pipeline_id}_{fit_dataset_id}_{best_score}_
        {pipeline_hash}_{fit_dataset_hash}"
        If pipeline not fitted 'fit_dataset_id' = None (or id unknown).
        The `best_score` available only after optimize step(s) if optimizer
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

    # @memory_profiler
    def predict(self, res, pipeline_id, dataset_id, subset_id='test',
                dirpath=None, **kwargs):
        """Predict and dump.

        Parameters
        ----------
        res : dict
            For compliance with producer logic.
        pipeline_id : str
            Pipeline identifier in `objects` to make prediction on
            `dataset_id__subset_id`
        dataset_id : str
            Dataset identifier in `objects`.
        subset_id : str, optional (default='test')
            Data subset identifier to predict on. If '', use full dataset.
        dirpath : str, optional(default=None)
            Absolute path dump dir or relative to 'self.project_dir' started
            with './'. If None,"self.project_path/results/models" is used.
        **kwargs: dict
        `   Additional kwargs to pass in dataset.dump_pred(**kwargs).

        Returns
        -------
        res : dict
            Unchanged input, for compliance with producer logic.

        Notes
        -----
        Resulted filename includes prefix:
        "{workflow_id}_{pipeline_id}_{fit_dataset_id}_{best_score}
        _{pipeline_hash}_{fit_dataset_hash}
        _{predict_dataset_id}_{predict_dataset_hash}"
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
                                dataset, dataset_id)
        fullpath = test.dump_pred(filepath, y_pred, **kwargs)
        self.logger.log(25, f"Save predictions for {dataset_id} to file:"
                            f"\n    {fullpath}")
        return res

    def plot(self, pipeline_id, dataset_id, hp_grid, optimizer_id, cls,
             **kwargs):
        self.logger.info("|__ GUI")

        pipeline = self.pipelines[pipeline_id]
        dataset = self.datasets[dataset_id]

        # Create base_plot
        # [alternative] better df
        # base_plot = targets
        # preserve original index
        # base_plot = pd.Series(index=raw.index.values,
        #                       data=np.arange(1, targets.shape[0]+1)).rename_axis(raw.index.name)
        # [deprecated] move to gui
        # base_plot = pd.DataFrame(index=raw.index.values,
        #                          data={target_name: np.arange(1, targets.shape[0]+1)}).rename_axis(raw.index.name)


        # we need only hp_grid flat:
        # either hp here in args
        # either combine tested hp for all optimizers if hp = {}
        runs = self._runs.get(f"{pipeline_id}|{dataset_id}", {})
        gui = cls(pipeline, dataset, runs, **kwargs)
        threading.Thread(target=gui.plot(), args=(), daemon=True).start()
        return

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
        pipeline : mlshell.Pipeline
            Pipeline, passed to `resolver`
        resolver : mlshell.Resolver
            Interface to resolve hp.
        dataset : mlshell.Dataset
            Dataset, passed to `resolver`.
        **resolve_params: : dict {hp_name: kwargs}
            Additional parameters to pass in `resolver.resolve(*arg,
            **resolve_params['hp_name'])` for specific hp.

        Returns
        -------
        hp: dict
            Resolved input hyper-parameters.

        """
        for hp_name, val in hp.items():
            if val == 'auto' or val == ['auto']:
                kwargs = resolve_params.get(hp_name, {})
                value = resolver.resolve(hp_name, pipeline, dataset, **kwargs)
                hp[hp_name] = value if val == 'auto' else [value]
        return hp

    def _resolve_scoring(self, scoring, pipeline):
        """Resolve scoring for grid search.

        Notes
        -----
        If None, 'accuracy' or 'r2' depends on estimator type.
        If list, resolve known metric id via `objects` and sklearn built-in.
        Otherwise passed unchanged.

        """
        if scoring is None:
            # Hard-code (default estimator could not exist).
            if pipeline.is_classifier():
                scoring = 'accuracy'
            elif pipeline.is_regressor():
                scoring = 'r2'
        elif isinstance(scoring, list):
            # Resolve if exist, else use sklearn built-in.
            for metric_id in scoring:
                scoring = {metric_id: self.objects[i] if i in self.objects
                           else sklearn.metrics.SCORERS[metric_id]
                           for i in metric_id}
        return scoring

    # ========================== dump/predict =================================
    def _prefix(self, res, dirpath, pipeline, pipeline_id,
                dataset=0, dataset_id=''):
        """Generate informative file prefix."""
        dataset_id_hash = hash(dataset)
        fit_dataset_id = getattr(pipeline, 'dataset_id', None)
        fit_dataset_hash = hash(self.objects.get(fit_dataset_id, 0))
        best_score = str(res.get('runs', {})
                            .get((pipeline_id, fit_dataset_id), {})
                            .get('best_score_', '')
                         ).lower()
        filepath = f"{dirpath}/{self.oid}_{pipeline_id}_{fit_dataset_id}_" \
                   f"{best_score}_{hash(pipeline)}_{fit_dataset_hash}_" \
                   f"{dataset_id}_{dataset_id_hash}"
        return filepath


if __name__ == '__main__':
    pass
