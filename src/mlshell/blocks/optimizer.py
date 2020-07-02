"""
The :mod:`mlshell.blocks.optimizer` contains example of 'Optimizer' class
proposes unified interface to work with underlying pipeline. Intended to be
used in `mlshell.Workflow`. For new pipeline formats no need to edit `Workflow`
 class, only update `Optimizer` interface logic.

As one of realization, `RandomizedSearchOptimizer` class provided to optimize
pipeline hyper-parameters with sklearn.model_selection.RandomizedSearchCV.
Some hp grid search no needs to fit whole pipeline steps, for efficient
searching special subclasses provided:
`ThresholdOptimizer` to brute force classification threshold for "positive"
class as separate optimization stage.
`KwargsOptimizer` to brute force arbitrary score function parameters as
separate optimization stage.

"""

import mlshell
import mlshell.custom
import sklearn
import pandas as pd
import copy
import tabulate
import jsbeautifier
import numpy as np


class Optimizer(object):  # SklearnOptimizerMixin
    """Unified optimizer interface.

    Implements interface to access arbitrary optimizer.
    Interface: dump_runs, update_best and all underlying
        optimizer object methods.

    Attributes
    ----------
    optimizer : sklearn optimizer
        Underlying optimizer.

    Notes
    -----
    Calling unspecified methods are redirected to underlying optimizer object.

    """

    def __init__(self):
        self.optimizer = None

    def __getattr__(self, name):
        """Redirect unknown methods to optimizer object."""
        def wrapper(*args, **kwargs):
            getattr(self.pipeline, name)(*args, **kwargs)
        return wrapper

    def __hash__(self):
        return str(self.optimizer)

    def update_best(self, prev):
        """Combine current optimizer results with previous stages.

        The logic of choosing the best run is set here. Currently best hp
        combination and corresponding estimator taken from the last stage.
        But if any hp brute force in more than one stage, more complicated rule
        is required to merge runs.

        Parameters
        ----------
        prev : dict
            Previous stage update_best output for some pipeline-data pair.
            Initially set to {}. See output format for all possible keys.

        Returns
        -------
        nxt : dict
            Result of merging runs on all optimization stages for some
            pipeline-data pair.
            {
                'params': list of dict
                    List of get_params() for all runs in stages.
                'best_params_' : dict
                    Best estimator `params`[`optimizer.best_index_`].
                'best_estimator_' : mlshell.Pipeline TODO: underlying?
                    Best estimator `optimizer.best_estimator_` or
                    `optimizer.estimator.set_params(**best_params_))` if
                    `best_estimator_` attribute is absent.
                'best_score_' : tuple
                    Best score ('scorer_id', `optimizer.best_score_`).
                    'scorer_id' get from str(`optimizer.refit`). If
                    best_score_ attribute is absent, ('', float('-inf')) used.
             }

        Notes
        -----
        `mlshell.Workflow` utilize:

        * 'best_estimator_' key to update pipeline in `objects`.
        * `params' in built-in plotter.
        * 'best_score_' in file name for dump/dump_pred.

        """
        curr = self.optimizer
        best_index_ = getattr(curr, 'best_index_', None)
        cv_results_ = getattr(curr, 'cv_results_', None)
        if best_index_ is None or cv_results_ is None:
            return prev

        params = cv_results_['params']
        best_params_ = params[best_index_]

        best_estimator_ = getattr(curr, 'best_estimator_', None)
        if best_estimator_ is None:
            best_estimator_ = curr.estimator.set_params(**best_params_)

        best_score_ = getattr(curr, 'best_score_', float('-inf'))
        if best_score_ is float('-inf'):
            scorer_id = ''
        else:
            scorer_id = str(getattr(curr, 'refit', ''))

        nxt = {
            'best_estimator_': best_estimator_,
            'best_params_': best_params_,
            'params': prev.get('params', []).extend(params),
            'best_score_': (scorer_id, best_score_),
        }
        return nxt

    def dump_runs(self, logger, dirpath, **kwargs):
        """Dump results.

        Parameters
        ----------
        logger : logging.Logger
            Logger to logs runs summary.
        dirpath : str
            Absolute path to dump dir.
        **kwargs : dict
            Additional kwargs to pass in low-level dumper.

        """
        self._pprint(logger, self.optimizer)
        if hasattr(self.optimizer, 'best_estimator_'):
            pipeline = getattr(self.optimizer, 'best_estimator_')
        else:
            pipeline = getattr(self.optimizer, 'estimator')
        runs = copy.deepcopy(self.optimizer.cv_results_)
        best_run_ind = self.optimizer.best_index_
        self._dump_runs(logger, dirpath, pipeline, runs, best_run_ind, **kwargs)
        return None

    def _pprint(self, logger, optimizer):
        """Pretty print optimizer results.

        Parameters
        ----------
        logger : logging.Logger
            Logger to logs runs summary.
        optimizer : sklearn optimizer
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
            logger.info(msg)
        logger.info(f"GridSearch best index:\n    {optimizer.best_index_}")
        logger.info(f"GridSearch time:\n    {runs_avg}")
        logger.log(25, f"CV best modifiers:\n"
                       f"    {jsb(str(best_modifiers))}")
        logger.info(f"CV best configuration:\n"
                    f"    {jsb(str(optimizer.best_params_))}")
        logger.info(f"CV best mean test score:\n"
                    f"    {getattr(optimizer, 'best_score_', 'n/a')}")
        return None

    def _find_modifiers(self, cv_results_):
        """Find varied hp."""
        modifiers = []
        for key, val in cv_results_.items():
            if not key.startswith('param_'):
                continue
            if isinstance(val, list):
                size = len(val)
            else:
                size = val.shape[0]
            if size > 1:
                modifiers.append(key)
        return modifiers

    def _dump_runs(self, logger, dirpath, pipeline, runs, best_run_ind, **kwargs):
        """Dumps grid search results.

        Parameters
        ----------
        logger : logging.Logger
            Logger to logs runs summary.
        dirpath : str
            Absolute path to dump dir.
        pipeline : mlshell.Pipeline TODO:
            Tuned pipeline.
        runs : dict or pandas.Dataframe
            Grid search results `optimizer.cv_results_`.
        best_run_ind : int
            Index of run with best score in `runs`.
        **kwargs : dict
            Additional kwargs to pass in low-level dumper.

        Notes
        -----
        In resulted file <timestamp>_runs.csv each row corresponds to run,
        columns:
        * 'id' random UUID for run (hp combination).
        * all pipeline parameters.
        * 'pipeline__type' regressor or classifier.
        * 'pipeline__estimator__name' estimator.__name__. TODO?
        * data
        * 'dataset__id'.
        * 'dataset__hash' pd.util.hash_pandas_object hash of data before split.
        * 'dataset_index' TODO: index for whole dataset?
        * conf.py id.  TODO: global conf id.

        """

        # TODO: runs_compliance needed? test _dump_runs
        # Get full params for each run.
        nums = len(runs['params'])
        lis = list(range(nums))
        # Clone params (not attached data).
        est_clone = sklearn.clone(pipeline)
        for i, param in enumerate(runs['params']):
            est_clone.set_params(**param)
            lis[i] = est_clone.get_params()
        # Too big to print.
        df = pd.DataFrame(lis)
        # Merge df with runs with replace, exchange args if don`t need replace.
        # cv_results consist suffix param_.
        param_labels = set(i for i in runs.keys() if 'param_' in i)
        if param_labels:
            other_labels = set(runs.keys())-param_labels
            update_labels = set(df.columns).intersection(other_labels)
            runs = pd.DataFrame(runs).drop(list(param_labels), axis=1, errors='ignore')
            df = pd.merge(df, runs, how='outer', on=list(update_labels),
                          left_index=True, right_index=True,
                          suffixes=('_left', '_right'))
        # pipeline
        # df = pd.DataFrame(res.cv_results_)
        # rows = df.shape[0]
        # df2 = pd.DataFrame([pipeline.get_params()])

        # unique id for param combination
        run_id_list = [str(uuid.uuid4()) for _ in range(nums)]

        df['id'] = run_id_list
        df['data__hash'] = self.data_hash
        df['pipeline__type'] = self.p['pipeline__type']
        df['pipeline__estimator__name'] = self.p['pipeline__estimator'].__class__.__name__
        df['gs__splitter'] = self.p['gs__splitter']
        df['data__split_train_size'] = self.p['data__split_train_size']
        df['data__source'] = jsbeautifier.beautify(str({
            key: self.p[key] for key in self.p if key.startswith('data')
        }))
        df['params__hash'] = self.p_hash
        # df=df.assign(**{'id':run_id_list, 'data__hash':data_hash_list, .. })

        # cast to string before dump and print, otherwise it is too long
        # alternative: json.loads(json.dumps(data)) before create df
        object_labels = list(df.select_dtypes(include=['object']).columns)
        df[object_labels] = df[object_labels].astype(str)

        filepath = '{}/{}_runs.csv'.format(dirpath, int(time.time()))
        with open(filepath, 'a', newline='') as f:
            df.to_csv(f, mode='a', header=f.tell() == 0, index=False, line_terminator='\n')
        logger.log(25, f"Save run(s) results to file:\n    {filepath}")
        logger.log(25, f"Best run id:\n    {run_id_list[best_run_ind]}")
        # alternative: to hdf(longer,bigger) hdfstore(can use as dict)
        # df.to_hdf(filepath, key='key', append=True, mode='a', format='table')

        # print (large only for debug)
        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #     self.logger.info('{}'.format(tabulate(df, headers='keys', tablefmt='psql')))

        # if not path.exists(filepath):
        #     df.to_csv(filepath, header=True, index=False)
        # else:
        #     df.to_csv(filepath, mode='a', header=False, index=False)

        # отдельная таблица для params
        # df2 = pd.DataFrame(self.p)
        # df2.to_csv('{}/params.csv'.format(self.project_path), index=False)



class RandomizedSearchOptimizer(Optimizer):
    def __init__(self, pipeline, hp_grid, scoring, mock=False, **kwargs):
        """

        Parameters
        ----------
        pipeline
        hp_grid
        scoring
        mock : bool, optional (default=False)
            If True, remove pipeline steps where hp path/sub-path not in
            hp_grid keys. For example, 'a_b_c' remains all 'a' sub-paths.
            Applied only if pipeline created with sklearn.pipeline.Pipeline.
            If hp_grid is {}, remain full pipeline.
        **kwargs

        Notes
        -----
        Be careful, if pipeline use `pass_custom` and mock set True without
        adding `pass_custom__kwargs` to hp_grid, corresponding custom score(s)
        will use last applied kwargs. The problem that it can be from another
        pipeline optimization, so better always remain pass_custom when mock.

        TODO: what with result pipeline? is it saved to self.objects?
        """
        super().__init__()
        n_iter = self._resolve_n_iter(kwargs.pop('n_iter', 10), hp_grid)
        if hp_grid and mock and hasattr(pipeline, 'steps'):
            pipeline = self._mock_pipeline(pipeline, hp_grid)
        self.optimizer = sklearn.model_selection.RandomizedSearchCV(
            pipeline, hp_grid, scoring=scoring, n_iter=n_iter, **kwargs)

    def fit(self, *args, **fit_params):
        self.optimizer.fit(*args, **fit_params)
        return None

    def _mock_pipeline(self, pipeline, hp_grid):
        """Remain steps only if in hp_grid."""
        r_step = []
        for step in pipeline.steps:
            if step[0] == 'pass_custom':
                print("Warning: Better always remain pass_custom when mock.")
            for hp_name in hp_grid:
                paths = hp_name.split('__')
                if step[0] == paths[0]:
                    r_step.append(step)
        mock_pipeline = sklearn.pipeline.Pipeline(steps=r_step)
        return mock_pipeline

    def _resolve_n_iter(self, n_iter, hp_grid):
        """Set number of runs in grid search."""
        # Calculate from hps ranges if 'n_iter' is None.
        if n_iter is not None:
            return n_iter
        try:
            # 1.0 if hp_grid = {}.
            n_iter = np.prod([len(i) if isinstance(i, list) else i.shape[0]
                              for i in hp_grid.values()])
        except AttributeError as e:
            raise ValueError("Distribution is used for hyperparameter grid, "
                             "specify 'runs' in gs_params.")
        return n_iter


class KwargsOptimizer(RandomizedSearchOptimizer):
    pass


class ThresholdOptimizer(RandomizedSearchOptimizer):
    """Threshold optimizer.

    Provide interface for separate optimize step to brute force threshold in
    classification without full pipeline fit.

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, x, y, **fit_params):
        optimizer = self.optimizer

        y_pred_proba, index = mlshell.custom.cross_val_predict(
            optimizer.estimator, x, y=y, fit_params=fit_params,
            groups=None, cv=optimizer.cv, method='predict_proba')

        optimizer.fit(y_pred_proba, y[index], **fit_params)

        # TODO: deprecated?
        if optimizer.refit:
            best_th_ = optimizer.best_params_[self.th_name]
            self.pipeline.set_params(**{self.th_name: best_th_})
            # Needs refit, otherwise not reproduce results.
            self.pipeline.fit(x, y, **fit_params)

        return None

    def runs_compliance(self, runs, runs_th_, best_index):
        """"Combine GS results to csv dump."""
        # runs.csv compliance
        # add param
        default_th = self.pipeline.get_params()['estimate__apply_threshold__threshold']
        runs['param_estimate__apply_threshold__threshold'] = np.full(len(runs['params']), fill_value=default_th)
        runs_th_['param_estimate__apply_threshold__threshold'] =\
            runs_th_.pop('param_threshold', np.full(len(runs_th_['params']),
                                                    fill_value=default_th))
        # update runs.params with param
        for run in runs['params']:
            run['estimate__apply_threshold__threshold'] = default_th
        for run in runs_th_['params']:
            run.update(runs['params'][best_index])
            run['estimate__apply_threshold__threshold'] = run.pop('threshold', default_th)
        # add all cv_th_ runs as separate rows with optimizer.best_params_ default values
        runs_df = pd.DataFrame(runs)
        runs_th_df = pd.DataFrame(runs_th_)
        sync_columns = [column for column in runs_th_df.columns if not column.endswith('time')]

        runs_df = runs_df.append(runs_th_df.loc[:, sync_columns], ignore_index=True)
        # replace Nan with optimizer.best_params_
        runs_df.fillna(value=runs_df.iloc[best_index], inplace=True)
        return runs_df
