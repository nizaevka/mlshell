"""
The :mod:`mlshell.blocks.optimizer` contains examples of `Optimizer` class to
to optimize pipeline hyper-parameters.

TODO:
dump_runs better here, cause we can change scheme

two otimizer, two otimize() function in workflow
best_params in pipeline mergable, also loked up hp_grid mergable
"""

import mlshell
import mlshell.custom
import sklearn
import pandas as pd


class SklearnOptimizerMixin(object):
    def __init__(self):
        self.optimizer = None

    def update_best(self, prev):
        """
            prev (dict): output from previous optimizers runs for this pipeline and data.
                Initially set to {}
        Note:
            Workflow need 'best_estimator_' key to update pipeline.
            Gui need list of params dict for each run.
            Dump predict/model need best_score_.
            More complicated logic on best_score_ or
        """
        curr = self.optimizer

        best_index_ = getattr(curr, 'best_index_', None)
        if not best_index_:
            return prev

        cv_results_ = getattr(curr, 'cv_results_')
        params = cv_results_['params']
        best_params_ = params[best_index_]
        best_estimator_ = getattr(curr, 'best_estimator_', None)
        if not best_estimator_:
            best_estimator_ = curr.estimator.set_params(**best_params_)

        best_score_ = getattr(curr, 'best_score_', float('-inf'))
        refit = getattr(curr, 'refit', '')
        score_name = refit if isinstance(refit, str) else ''
        next_ = {
            'best_estimator_': best_estimator_,
            'best_params_': best_params_,
            'params': prev['params'].extend(params),
            'best_score_': (score_name, best_score_),
        }
        return next_

    def dump_runs(self, logger, filepath):
        self._pretty_print(logger, self.optimizer)
        init_pipeline = getattr(self.optimizer, 'estimator')
        pipeline = getattr(self.optimizer, 'best_estimator_', init_pipeline)
        runs = copy.deepcopy(self.optimizer.cv_results_)
        best_run_index = self.optimizer.best_index_
        self._dump_runs(logger, filepath, pipeline, runs, best_run_index)

    def _dump_runs(self, logger, filepath, pipeline, runs, best_run_index):
        """Dumps grid search results in <timestamp>_runs.csv

        Args:
            runs (dict or pd.Dataframe): contain GS results.
            best_run_index (int): best score run index.

        Note:
            _runs.csv contain columns:

                * all estimator parameters.
                * 'id' random UUID for one run (hp combination).
                * 'data__hash' pd.util.hash_pandas_object hash of data before split.
                * 'params__hash' user params md5 hash (cause of function memory address will change at each workflow).
                * 'pipeline__type' regressor or classifier.
                * 'pipeline__estimator__name' estimator.__name__.
                * 'gs__splitter'.
                * 'data__split_train_size'.
                * 'data__source' params['data'].
        """
        print('OK')
        return

        # TODO: runs_comppliance needed? test _dump_runs
        logger.info("\u25CF \u25B6 DUMP RUNS")
        # get full params for each run
        nums = len(runs['params'])
        lis = list(range(nums))
        est_clone = sklearn.clone(pipeline)  # not clone attached data, only params
        for i, param in enumerate(runs['params']):
            est_clone.set_params(**param)
            lis[i] = est_clone.get_params()
        df = pd.DataFrame(lis)  # too big to print
        # merge df with runs with replace (exchange args if don`t need replace)
        # cv_results consist suffix param_
        param_labels = set(i for i in runs.keys() if 'param_' in i)
        if param_labels:
            other_labels = set(runs.keys())-param_labels
            update_labels = set(df.columns).intersection(other_labels)
            runs = pd.DataFrame(runs).drop(list(param_labels), axis=1, errors='ignore')
            df = pd.merge(df, runs,
                          how='outer', on=list(update_labels), left_index=True, right_index=True,
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

        with open(filepath, 'a', newline='') as f:
            df.to_csv(f, mode='a', header=f.tell() == 0, index=False, line_terminator='\n')
        logger.log(25, f"Save run(s) results to file:\n    {filepath}")
        logger.log(25, f"Best run id:\n    {run_id_list[best_run_index]}")
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

    # [deprecated]
    # def distribution_compliance(self, res, hp_grid):
    #     for name, vals in hp_grid.items():
    #         # check if distribution in hp_grid
    #         if not hasattr(type(vals), '__iter__'):
    #             # there are masked array in res
    #             hp_grid[name] = pd.unique(np.ma.getdata(res[f'param_{name}']))
    #     return hp_grid

    def _find_modifiers(self, cv_results_):
        # find varied hp
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

    def _pretty_print(self, logger, optimizer):
        """Pretty print."""

        # [deprecated] excessive, not work with distributions
        # self.logger.info('hp grid:\n    {}'.format(jsbeautifier.beautify(str(hp_grid))))
        modifiers = self._find_modifiers(optimizer.cv_results_)

        param_modifiers = set('param_'+i for i in modifiers)
        # outputs
        runs_avg = {'mean_fit_time': optimizer.cv_results_['mean_fit_time'].mean(),
                    'mean_score_time': optimizer.cv_results_['mean_score_time'].mean()}
        df = pd.DataFrame(optimizer.cv_results_)[[key for key in optimizer.cv_results_ if key in param_modifiers
                                                  or 'mean_train' in key or 'mean_test' in key]]
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            # logger.debug('{}'.format(df.head()))
            logger.info('{}'.format(tabulate.tabulate(df, headers='keys', tablefmt='psql')))
        # Alternative: df.to_string()

        logger.info('GridSearch best index:\n    {}'.format(optimizer.best_index_))
        logger.info('GridSearch time:\n    {}'.format(runs_avg))
        logger.log(25, 'CV best modifiers:\n'
                       '    {}'.format(jsbeautifier.beautify(str({key: optimizer.best_params_[key]
                                                                       for key in modifiers
                                                                       if key in optimizer.best_params_}))))
        logger.info('CV best configuration:\n'
                    '    {}'.format(jsbeautifier.beautify(str(optimizer.best_params_))))
        logger.info('CV best mean test score:\n'
                    '    {}'.format(optimizer.__dict__.get('best_score_', 'n/a')))  # not exist if refit callable
        # [deprecated] long ago
        # Alternative: nested dic to MultiIndex df
        # l = res.cv_results_['mean_fit_time'].shape[0]
        # dic = dict({'index':np.arange(l, dtype=np.float), 'train_score':res.cv_results_['mean_train_score'],
        #              'test_score': res.cv_results_['mean_test_score']},
        #              **{key:res.cv_results_[key] for key in res.cv_results_ if 'param_' in key})
        # Example:
        # dic = {'a': list(range(10)), 'b': {'c': list(range(10)), 'd': list(range(10))}}
        # dic_flat = {}
        # dic_flatter(dic, dic_flat)
        # pd.DataFrame(dic_flat)

    def _resolve_n_iter(self, n_iter, hp_grid):
        """Set number of runs in grid search."""
        # calculate from hps ranges if user 'gs__runs' is not given
        if n_iter is not None:
            return n_iter
        try:
            # 1.0 if hp_grid={}
            n_iter = np.prod([len(i) if isinstance(i, list) else i.shape[0]
                              for i in hp_grid.values()])
        except AttributeError as e:
            raise ValueError("Distribution is used for hyperparameter grid, "
                             "specify 'runs' in gs_params.")
        return n_iter


class RandomizedSearchOptimizer(SklearnOptimizerMixin):
    def __init__(self, pipeline, hp_grid, scoring, **kwargs):
        super().__init__()
        n_iter = self._resolve_n_iter(kwargs.pop('n_iter', 10), hp_grid)
        self.optimizer = sklearn.model_selection.RandomizedSearchCV(
            pipeline, hp_grid, scoring=scoring, n_iter=n_iter, **kwargs)

    def fit(self, *args, **fit_params):
        self.optimizer.fit(*args, **fit_params)
        return None


class MockOptimizer(SklearnOptimizerMixin):
    """Threshold optimizer.

    Provide interface for separate optimize step to brute force threshold in
    classification without full pipeline fit.

    """
    def __init__(self, pipeline, hp_grid, scoring, **kwargs):
        """

        Parameters
        ----------
        pipeline
        hp_grid
        scoring
        hp_mock : list of str
            Pipeline hp to mock for brute force.
        **kwargs


        """
        super().__init__()
        n_iter = self._resolve_n_iter(kwargs.pop('n_iter', 10), hp_grid)
        mock_pipeline = self._resolve_pipeline(pipeline, hp_grid)
        self.optimizer = sklearn.model_selection.RandomizedSearchCV(
            mock_pipeline, hp_grid, scoring=scoring, n_iter=n_iter, **kwargs)

    def _resolve_pipeline(self, pipeline, hp_grid):
        """Reproduce pipeline steps structure."""
        # Default (if not hp_grid).
        mock_pipeline = pipeline
        params = pipeline.get_params()
        for hp_name in hp_grid:  # 'a__b__c'
            # Copy whole original step, contain hp_name last parameter
            # and recover upstream name structure (envelop all except last).
            # If only one subname, copy whole pipeline.
            lis = hp_name.split('__')  # ['a', 'b', 'c']
            step_name = '__'.join(lis[:-1])  # ['a', 'b']

            if step_name:
                for subname in lis[-2::-1]:  # ['b', 'a']
                    # Check if subname already exist.

                    mock_pipeline = sklearn.pipeline.Pipeline(
                        steps=[(subname, params[step_name])])
            else:
                mock_pipeline = pipeline
        return mock_pipeline

    def fit(self, x, y, **fit_params):
        optimizer = self.optimizer

        # First do it simple, test, then complicate.
        # TODO: maybe move out in workflow, so unify with brut custom_score
        #     method = x, predict_proba(for threshold), predict(for scorer)
        #     Invokes the passed method name of the passed estimator.
        #     but i not sure that it is possible: in some case mean score, in some OOF
        #     If so i can not even redefine fit
        # TODO: Everything with data+pipeline should separate from data and pipeline
        #    resolver also should be in utils.sklearn. Workflow get as params this utils
        #    pipeline.resolve should be in pipeline, but resolver in utils.sklearn with validator

        y_pred_proba, _, y_true = mlshell.custom.cross_val_predict(
            optimizer.estimator,
            x, y=y, fit_params=fit_params,
            groups=None, cv=optimizer.cv, method='predict_proba')

        optimizer.fit(y_pred_proba, y_true, **fit_params)

        # [deprecated]
        # best_th_ = optimizer.best_params_['threshold']
        # runs_th_ = copy.deepcopy(optimizer.cv_results_)

        # best_run_index = len(runs['params']) + optimizer.best_index_
        # # better make in dump runs in auto regime (could be problem if CV differs)
        # runs = self.runs_compliance(runs, runs_th_, optimizer.best_index_)

        # self.best_params_['estimate__apply_threshold__threshold'] = best_th_
        # self.modifiers.append('estimate__apply_threshold__threshold')
        # self.p['gs__hp_grid']['estimate__apply_threshold__threshold'] = th_range

        if optimizer.refit:
            best_th_ = optimizer.best_params_[self.th_name]
            self.pipeline.set_params(**{self.th_name: best_th_})
            #  need refit, otherwise not reproduce results
            self.pipeline.fit(x, y, **fit_params)

        # recover sklearn optimizer structure
        self.__dict__.update(self.optimizer.__dict__)
        return self

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
