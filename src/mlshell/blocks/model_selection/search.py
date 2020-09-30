"""
The :mod:`mlshells.model_selection.search` includes utilities to
optimize hyper-parameters.

:class:`mlshell.model_selection.Optimizer` class proposes unified interface to
arbitrary optimizer. Intended to be used in :class:`mlshell.Workflow` . For new
optimizer formats no need to edit `Workflow` class, just adapt in compliance to
interface.

:class:`mlshell.model_selection.RandomizedSearchOptimizer` contains
:class:`sklearn.model_selection.RandomizedSearchCV` implementation.
:class:`mlshell.model_selection.MockOptimizer` subclass provides efficient
brute force prediction-related parameters as separate optimize step. For
example: classification threshold or scorer function kwargs don`t need whole
pipeline refit to probe.

"""

import copy
import os
import platform
import time
import uuid

import jsbeautifier
import mlshell
import numpy as np
import pandas as pd
import sklearn
import tabulate

__all__ = ['Optimizer', 'RandomizedSearchOptimizer', 'MockOptimizer']


class Optimizer(object):
    """Unified optimizer interface.

    Implements interface to access arbitrary optimizer.
    Interface: dump_runs, update_best and all underlying optimizer methods.

    Attributes
    ----------
    optimizer : :class:`sklearn.model_selection.BaseSearchCV`
        Underlying optimizer.

    Notes
    -----
    Calling unspecified methods are redirected to underlying optimizer object.

    """

    def __init__(self):
        self.optimizer = None

    def __hash__(self):
        return str(self.optimizer)

    def __getattr__(self, name):
        """Redirect unknown methods to optimizer object."""
        def wrapper(*args, **kwargs):
            getattr(self.optimizer, name)(*args, **kwargs)
        return wrapper

    def __getstate__(self):
        # Allow pickle.
        return self.__dict__

    def __setstate__(self, d):
        # Allow unpickle.
        self.__dict__ = d

    def update_best(self, prev):
        """Combine results from multi-stage optimization.

        The logic of choosing the best run is set here. Currently best hp
        combination and corresponding estimator taken from the last stage.
        But if any hp brute force in more than one stage, more complicated rule
        is required to merge runs.

        Parameters
        ----------
        prev : dict
            Previous stage ``update_best`` output for some pipeline-data pair.
            Initially set to {}. See ``update_best`` output format.

        Returns
        -------
        nxt : dict
            Result of merging runs on all optimization stages for some
            pipeline-data pair: {

                'params': list of dict
                    List of ``cv_results_['params']`` for all runs in stages.
                'best_params_' : dict
                    Best estimator tuned params from all optimization stages.
                'best_estimator_' : :mod:`sklearn` estimator
                    Best estimator ``optimizer.best_estimator_`` if exist, else
                    ``optimizer.estimator.set_params(**best_params_))`` (
                    if not 'refit' is True).
                'best_score_' : tuple
                    Best score ``('scorer_id', optimizer.best_score_)`` , where
                    ``scorer_id=str(optimizer.refit)``. If best_score_  is
                    absent, ``('', float('-inf'))`` used.

             }

        Notes
        -----
        :class:`mlshell.Workflow` utilize:

        * 'best_estimator_' key to update pipeline in ``objects``.
        * 'params' in built-in plotter.
        * 'best_score_' in dump/dump_pred for file names.

        """
        curr = self.optimizer
        best_index_ = getattr(curr, 'best_index_', None)
        cv_results_ = getattr(curr, 'cv_results_', None)
        if best_index_ is None or cv_results_ is None:
            return prev

        # Only modifiers.
        params = cv_results_['params']
        best_params_ = params[best_index_]
        best_estimator_ = getattr(curr, 'best_estimator_', None)
        if best_estimator_ is None:
            # If not 'refit'.
            best_estimator_ = curr.estimator.set_params(**best_params_)
        params_init = best_estimator_.get_params()
        params_full = [{**params_init, **p} for p in params]

        best_score_ = getattr(curr, 'best_score_', float('-inf'))
        if best_score_ is float('-inf'):
            scorer_id = ''
        else:
            scorer_id = str(getattr(curr, 'refit', ''))

        nxt = {
            'best_estimator_': best_estimator_,
            'best_params_': {**prev.get('best_params_', {}), **best_params_},
            'params': [*prev.get('params', []), *params_full],
            'best_score_': (scorer_id, best_score_),
        }
        return nxt

    def dump_runs(self, logger, dirpath, pipeline, dataset, **kwargs):
        """Dump results.

        Parameters
        ----------
        logger : :class:`logging.Logger`
            Logger.
        dirpath : str
            Absolute path to dump dir.
        pipeline : :class:`mlshell.Pipeline`
            Pipeline used for optimizer.fit.
        dataset : :class:`mlshell.Dataset`
            Dataset used for optimizer.fit.
        **kwargs : dict
            Additional kwargs to pass in low-level dump function.

        Notes
        -----
        Resulted file name ``<timestamp>_runs.csv``. Each row corresponds to
        run, columns names:

        * 'id' random UUID for run (hp combination).
        * All pipeline parameters.
        * Grid search output ``runs`` keys.
        * Pipeline info: 'pipeline__id', 'pipeline__hash', 'pipeline__type'.
        * Dataset info: 'dataset__id', 'dataset__hash'.

        Hash could alter when interpreter restarted, because of address has
        changed for some underlying function.
        """
        runs = copy.deepcopy(self.optimizer.cv_results_)
        best_ind = self.optimizer.best_index_

        self._pprint(logger, self.optimizer)
        self._dump_runs(logger, dirpath, pipeline, dataset, runs, best_ind,
                        **kwargs)
        return None

    def _pprint(self, logger, optimizer):
        """Pretty print optimizer results.

        Parameters
        ----------
        logger : :class:`logging.Logger`
            Logger.
        optimizer : :class:`sklearn.model_selection.BaseSearchCV`
            Underlying optimizer.

        """
        jsb = jsbeautifier.beautify
        modifiers = self._find_modifiers(optimizer.cv_results_)
        param_modifiers = set(f'param_{i}' for i in modifiers)
        best_modifiers = {key: optimizer.best_params_[key] for key in modifiers
                          if key in optimizer.best_params_}
        runs_avg = {
            'mean_fit_time': optimizer.cv_results_['mean_fit_time'].mean(),
            'mean_score_time': optimizer.cv_results_['mean_score_time'].mean()
        }
        useful_keys = [key for key in optimizer.cv_results_
                       if key in param_modifiers
                       or 'mean_train' in key or 'mean_test' in key]
        df = pd.DataFrame(optimizer.cv_results_)[useful_keys]

        with pd.option_context('display.max_rows', None,
                               'display.max_columns', None):
            msg = tabulate.tabulate(df, headers='keys', tablefmt='psql')
            logger.info(f"")
            logger.info(f"CV results:\n{msg}")
        logger.info(f"GridSearch time:\n    {runs_avg}")
        logger.info(f"GridSearch best index:\n    {optimizer.best_index_}")
        logger.info(f"CV best modifiers:\n"
                    f"    {jsb(str(best_modifiers))}")
        logger.log(25, f"CV best configuration:\n"
                       f"    {jsb(str(optimizer.best_params_))}")
        logger.log(25, f"CV best mean test score "
                       f"{getattr(optimizer, 'refit', '')}:\n"
                       f"    {getattr(optimizer, 'best_score_', 'n/a')}")
        return None

    def _find_modifiers(self, cv_results_):
        """Find varied hp."""
        # [alternative] from optimizer.param_distributions, but would not work
        #   if sampler set. So better find out from already sampled.
        # Remove 'param_' prefix.
        modifiers = []
        for key, val in cv_results_.items():
            if not key.startswith('param_'):
                continue
            # Internally always masked array.
            # Unmask, check if not homo.
            val = val.data
            for el in val:
                if el != val[0]:
                    modifiers.append(key.replace('param_', '', 1))
                    break
            # [deprecated] work, but deprecation warning
            # also can`t use unique(object are not comparable).
            # if not np.all(val == val[0]):
            #     modifiers.append(key.replace('param_', '', 1))
        return modifiers

    def _dump_runs(self, logger, dirpath, pipeline, dataset, runs, best_ind,
                   **kwargs):
        """Dumps grid search results.

        Parameters
        ----------
        logger : :class:`logging.Logger`
            Logger.
        dirpath : str
            Absolute path to dump dir.
        pipeline : :class:`mlshell.Pipeline`
            Pipeline used for optimizer.fit.
        dataset : :class:`mlshell.Dataset`
            Dataset used for optimizer.fit.
        runs : dict or :class:`pandas.Dataframe`
            Grid search results ``optimizer.cv_results_`` .
        best_ind : int
            Index of run with best score in `runs`.
        **kwargs : dict
            Additional kwargs to pass in low-level dumper.

        """
        # Create df with runs pipeline params.
        df = pd.DataFrame(self._runs_hp(runs, pipeline))
        # Add results
        df = self._runs_results(df, runs)
        # Add unique id.
        id_list = [str(uuid.uuid4()) for _ in range(df.shape[0])]
        df['id'] = id_list
        # Add pipeline info.
        df['pipeline__id'] = pipeline.oid
        df['pipeline__hash'] = hash(pipeline)
        df['pipeline__type'] = 'regressor' if pipeline.is_regressor()\
            else 'classifier'
        # Add dataset info.
        df['dataset__id'] = dataset.oid  # section__config__subset
        df['dataset__hash'] = hash(dataset)

        # Cast 'object' type to str before dump, otherwise it is too long.
        object_labels = list(df.select_dtypes(include=['object']).columns)
        df[object_labels] = df[object_labels].astype(str)
        # Dump.
        filepath = f"{dirpath}/{int(time.time())}_{platform.system()}_runs.csv"
        if "PYTEST_CURRENT_TEST" in os.environ:
            if 'float_format' not in kwargs:
                kwargs['float_format'] = '%.8f'
        with open(filepath, 'a', newline='') as f:
            df.to_csv(f, mode='a', header=f.tell() == 0, index=False,
                      line_terminator='\n', **kwargs)
        logger.log(25, f"Save run(s) results to file:\n    {filepath}")
        logger.log(25, f"Best run id:\n    {id_list[best_ind]}")

    def _runs_hp(self, runs, pipeline):
        # Make list of get_params() for each run.
        lis = list(range(len(runs['params'])))
        # Clone params (not attached data).
        est_clone = sklearn.clone(pipeline.pipeline)
        for i, param in enumerate(runs['params']):
            est_clone.set_params(**param)
            lis[i] = est_clone.get_params()
        return lis

    def _runs_results(self, df, runs):
        """Add output columns."""
        # For example: mean_test_score/mean_fit_time/...
        # Hp already in df, runs (cv_results) consists suffix param_ for
        # modifiers. For columns without suffix: merge with replace.
        # If no params_labels, anyway one combination checked.
        param_labels = set(i for i in runs.keys() if 'param_' in i)
        other_labels = set(runs.keys())-param_labels
        update_labels = [name for name in df.columns if name in other_labels]
        runs = pd.DataFrame(runs).drop(list(param_labels), axis=1,
                                       errors='ignore')
        df = pd.merge(df, runs, how='outer', on=list(update_labels),
                      left_index=True, right_index=True,
                      suffixes=('_left', '_right'))
        return df


class RandomizedSearchOptimizer(Optimizer):
    """Wrapper around :class:`sklearn.model_selection.RandomizedSearchCV`.

    Parameters
    ----------
    pipeline: :mod:`sklearn` estimator
        See corresponding argument for
        :class:`sklearn.model_selection.RandomizedSearchCV`.
    hp_grid: dict
        See corresponding argument for
        :class:`sklearn.model_selection.RandomizedSearchCV`.
        Only `dict` type for ``hp_grid`` currently supported.
    scoring:  string, callable, list/tuple, dict, optional (default=None)
        See corresponding argument for
        :class:`sklearn.model_selection.RandomizedSearchCV`.
    **kwargs : dict
        Kwargs for :class:`sklearn.model_selection.RandomizedSearchCV`.
        If kwargs['n_iter']=None, replaced with number of hp combinations
        in ``hp_grid`` ("1" if only distributions found or empty).

    """
    def __init__(self, pipeline, hp_grid, scoring, **kwargs):
        super().__init__()
        n_iter = self._resolve_n_iter(kwargs.pop('n_iter', 10), hp_grid)
        self.optimizer = sklearn.model_selection.RandomizedSearchCV(
            pipeline, hp_grid, scoring=scoring, n_iter=n_iter, **kwargs)

    def _resolve_n_iter(self, n_iter, hp_grid):
        """Set number of runs in grid search."""
        # Calculate from hps ranges if 'n_iter' is None.
        if n_iter is not None:
            return n_iter

        # 1.0 if hp_grid = {} or distributions.
        n_iter = 1
        for i in hp_grid.values():
            try:
                n_iter *= len(i)
            except AttributeError:
                # Either distribution, or single value(error raise in sklearn).
                continue
        return n_iter


class MockOptimizer(RandomizedSearchOptimizer):
    """Threshold optimizer.

    Provides interface to efficient brute force prediction-related parameters
    in separate optimize step. For example: classification threshold or score
    function kwargs. 'MockOptimizer' avoids pipeline refit for such cases.
    Internally :class:`mlshell.model_selection.cross_val_predict` called with
    specified ``method`` and hp optimized on output prediction.

    Parameters
    ----------
    pipeline: :mod:`sklearn` estimator
        See corresponding argument for
        :class:`sklearn.model_selection.RandomizedSearchCV`.
    hp_grid : dict
        Specify only ``hp`` supported mock optimization: should not depends on
        prediction. If {}, :class:`mlshell.custom.MockClassifier` or
        :class:`mlshell.custom.MockRegressor` used for compliance.
    scoring: string, callable, list/tuple, dict, optional (default=None)
        See corresponding argument in :class:`sklearn.model_selection.
        RandomizedSearchCV`.
    method : str {'predict_proba', 'predict'}, optional (default='predict')
        Set ``predict_proba`` if classifier supported and if any metric
        ``needs_proba``. See corresponding argument for
        :class:`mlshell.model_selection.cross_val_predict`.
    **kwargs : dict
        Kwargs for :class:`sklearn.model_selection.RandomizedSearchCV`.
        If kwargs['n_iter']=None, replaced with number of hp combinations
        in ``hp_grid``.

    Notes
    -----
    To brute force threshold, set method to 'predict_proba'.
    To brute force scorer kwargs alone could be 'predict' or 'predict_proba'
    depends on if scoring needs probabilities.

    """
    def __init__(self, pipeline, hp_grid, scoring,
                 method='predict', **kwargs):
        self._check_method(method, scoring)
        self.method = method
        self.pipeline = pipeline
        mock_pipeline = self._mock_pipeline(pipeline, hp_grid)
        super().__init__(mock_pipeline, hp_grid, scoring, **kwargs)

    def fit(self, x, y, **fit_params):
        optimizer = self.optimizer

        # y_pred depends on method.
        y_pred, index = mlshell.model_selection.cross_val_predict(
            self.pipeline, x, y=y, fit_params=fit_params,
            groups=None, cv=optimizer.cv, method=self.method)
        # Multioutput compliance (otherwise error length x,y inconsistent).
        if self.method == 'predict_proba':
            pseudo_x = y_pred if not isinstance(y_pred, list) \
                else np.dstack(y_pred)  # list(zip(*y_pred))
        else:
            pseudo_x = y_pred
        optimizer.fit(pseudo_x, y[index], **fit_params)

        optimizer = self._mock_optimizer(optimizer, x, y, fit_params)
        self.optimizer = optimizer
        return None

    def _check_method(self, method, scoring):
        """Check method arg."""
        if isinstance(scoring, dict):
            for key, val in scoring.items():
                # scorer could be from sklearn SCORERS => no needs_proba attr.
                typ = sklearn.metrics._scorer._ProbaScorer
                if method == 'predict':
                    if isinstance(val, typ) \
                            or getattr(val, 'needs_proba', False):
                        raise ValueError(f"MockOptimizer: scoring {key} needs "
                                         f"method=predict_proba in gs_params,"
                                         f" but predict use.")
        return

    def _mock_pipeline(self, pipeline, hp_grid):
        """Remain steps only if in hp_grid.

        Notes
        -----
        if 'a_b_c' in hp_grid, remains 'a_b' sub-path: copy 'a', remove unused
        paths.


        If resulted mock pipeline has no predict method, added:
        ('estimate_del',
            :class:`mlshell.model_selection.prediction.MockClassifier` or
            :class:`mlshell.model_selection.prediction.MockRegressor`
        ).

        Always remains 'pass_custom' step. If pipeline use `pass_custom` and
        `pass_custom__kwargs` not in hp_grid, corresponding custom score(s)
        will use last applied kwargs. The problem that it can be from
        pipeline optimization on another dataset.

        """
        # params = pipeline.get_params()
        # Set of used path subpaths.
        # Always remain pass_custom when mock.
        paths = {'pass_custom'}
        if self.method == 'predict_proba':
            paths.add('estimate')
            paths.add('estimate__apply_threshold')
        for i in hp_grid:
            lis = i.split('__')
            for j in range(len(lis)):
                paths.add('__'.join(lis[:j+1]))

        def recursion(steps, path):
            del_inds = []
            for ind, substep in enumerate(steps):
                subpath = f'{path}__{substep[0]}' if path else substep[0]
                if subpath not in paths:
                    # Delete.
                    del_inds.append(ind)
                elif subpath in hp_grid:
                    # Remain full element, no need to level up, otherwise
                    # will remove all inside the step
                    # for example 'estimate'.
                    continue
                elif hasattr(substep[1], 'steps'):
                    if substep[1].steps:
                        # Level down.
                        recursion(substep[1].steps, subpath)
                        if not substep[1].steps:
                            # if remove all from steps.
                            del_inds.append(ind)
                    else:
                        # Delete full element (empty steps).
                        del_inds.append(ind)
                else:
                    # Remain full element(no level up and in paths)
                    pass
            for ind in del_inds[::-1]:
                del steps[ind]
            return

        if hasattr(pipeline, 'steps'):
            r_steps = copy.deepcopy(pipeline.steps)
        else:
            # Can`t separate any hp to brut force.
            raise AttributeError(
                "MockOptimizer: pipeline should contain steps, "
                "otherwise efficient optimization not possible."
            )
        recursion(r_steps, '')

        flag = False
        if not r_steps:
            # No estimator save from hp_grid (for example empty).
            flag = True
        elif r_steps:
            mock_pipeline = sklearn.pipeline.Pipeline(steps=r_steps)
            if not hasattr(mock_pipeline, 'predict'):
                # No estimator.
                flag = True

        if flag:
            # Add MockEstimator step (non modifier, not change cv_results_).
            mock = self._mock_estimator(pipeline)
            r_steps.append(('estimate_del', mock))
        mock_pipeline = sklearn.pipeline.Pipeline(steps=r_steps)
        return mock_pipeline

    def _mock_estimator(self, pipeline):
        """Mock last step estimator."""
        if sklearn.base.is_regressor(estimator=pipeline):
            mock = mlshell.model_selection.prediction.MockRegressor()
        elif self.method == 'predict':
            mock = mlshell.model_selection.prediction.MockClassifier()
        elif self.method == 'predict_proba':
            # If already in, should have remained.
            raise ValueError("MockOptimizer: needs ThresholdClassifier "
                             "in pipeline steps.")
            # TODO: somehow resolved here, additional kwargs.
            # mock = mlshell.model_selection.prediction.ThresholdClassifier(
            #     pipeline.params, pipeline.threshold)
        else:
            raise ValueError(f"Unknown method {self.method}")
        return mock

    def _mock_optimizer(self, optimizer, x, y, fit_params):
        if hasattr(optimizer, 'refit') and optimizer.refit:
            self.pipeline.set_params(**optimizer.best_params_)
            # Needs refit, otherwise not reproduce results.
            self.pipeline.fit(x, y, **fit_params)

        # Recover structure as if fit original pipeline.
        for attr, val in {'estimator': self.pipeline,
                          'best_estimator_': self.pipeline}.items():
            try:
                setattr(optimizer, attr, val)
            except AttributeError:
                continue
        return optimizer


if __name__ == '__main__':
    pass
