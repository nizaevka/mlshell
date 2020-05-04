"""TODO: make as shell over RandomizedSearchCV"""

import sklearn


class Optimizer(object):
    def __init__(self, pipeline, **kwargs):
        # if not hp_grid:
        #      hp_grid = {}

    def fit(self, x, y, **kwargs):
        """Tune hp on train by cv."""
        # TODO: Add mandatory default keys with warnings.
        self.check_gs_keys()
        # param, fold -> fit(fold_train) -> predict(fold_test) -> score for params
        self.scorers = self.metrics_to_scorers(self.p['metrics'], self.p['gs__metrics'])
        self.refit = self.get_refit(self.p)
        hp_grid = self.get_hp_grid(self.p['gs__hp_grid'])
        scoring, th_range, refit_updated = self.get_scoring(self.refit)
        n_iter = self.get_n_iter()
        pre_dispatch = self.get_pre_dispatch()

        # optimize score
        optimizer = sklearn.model_selection.RandomizedSearchCV(
            self.estimator, hp_grid, scoring=scoring, n_iter=n_iter,
            n_jobs=self.p['gs__n_jobs'], pre_dispatch=pre_dispatch,
            refit=refit_updated, cv=self.cv(), verbose=self.p['gs__verbose'], error_score=np.nan,
            return_train_score=True)
        optimizer.fit(self.x_train, self.y_train, **self.p['pipeline__fit_params'])
        self.estimator = optimizer.best_estimator_
        self.best_params_ = optimizer.best_params_
        best_run_index = optimizer.best_index_
        if 'pass_custom__kw_args' in self.best_params_:
            self.default_custom_kw_args = self.best_params_['pass_custom__kw_args']
        self.distribution_compliance(optimizer.cv_results_, hp_grid)
        self.modifiers = self.find_modifiers(hp_grid)
        runs = copy.deepcopy(optimizer.cv_results_)
        # nice print
        self.gs_print(optimizer, self.modifiers)

        # optimize threshold if necessary
        if self.p['pipeline__type'] == 'classifier' and (self.p['th__strategy'] == 1 or self.p['th__strategy'] == 3):
            self.logger.info("\u25CF \u25B6 GRID SEARCH CLASSIFIER THRESHOLD")
            scoring = self.scorers
            th_range, predict_proba, y_true = self.calc_th_range(th_range)
            optimizer_th_ = sklearn.model_selection.RandomizedSearchCV(
                mlshell.custom.ThresholdClassifier(self.classes_, self.pos_label_ind,
                                                   self.pos_label, self.neg_label),
                {'threshold': th_range}, n_iter=th_range.shape[0],
                scoring=scoring,
                n_jobs=1, pre_dispatch=2, refit=self.refit, cv=self.cv(),
                verbose=1, error_score=np.nan, return_train_score=True).fit(predict_proba, y_true,
                                                                            **self.p['pipeline__fit_params'])
            best_th_ = optimizer_th_.best_params_['threshold']
            runs_th_ = copy.deepcopy(optimizer_th_.cv_results_)
            best_run_index = len(runs['params']) + optimizer_th_.best_index_
            self.distribution_compliance(optimizer.cv_results_, hp_grid)

            runs = self.runs_compliance(runs, runs_th_, optimizer.best_index_)
            self.best_params_['estimate__apply_threshold__threshold'] = best_th_
            self.modifiers.append('estimate__apply_threshold__threshold')
            self.p['gs__hp_grid']['estimate__apply_threshold__threshold'] = th_range
            self.gs_print(optimizer_th_, ['threshold'])

            #  need refit, otherwise not reproduce results
            self.estimator.set_params(**{'estimate__apply_threshold__threshold': best_th_})

            # refit with threshold (only after update best_params)
            # self.estimator.set_params(**self.best_params_)  # the same as just threshold
            # # assert np.array_equal(self.y_train.values, y_true)
            self.estimator.fit(self.x_train, self.y_train, **self.p['pipeline__fit_params'])

        return runs, best_run_index

    def check_gs_keys(self):
        # TODO:
        pass

    def metrics_to_scorers(self, metrics, gs_metrics):
        """Make from scorers from metrics

        Args:
            metrics (dict): {'name': (sklearn metric object, bool greater_is_better), }
            gs_metrics (sequence of str): metrics names to use in gs.

        Returns:
            scorers (dict): {'name': sklearn scorer object, }

        Note:
            if 'gs__metric_id' is None, estimator default will be used.

        """
        scorers = {}
        if not gs_metrics:
            # need to set explicit, because always need not None 'refit' name
            # can`t extract estimator built-in name, so use all from validation metrics
            gs_metrics = metrics.keys()

        for name in gs_metrics:
            if name in metrics:
                metric = metrics[name]
                if isinstance(metric[0], str):
                    # convert to callable
                    # ignore kw_args (built-in `str` metrics has hard-coded kwargs)
                    scorers[name] = sklearn.metrics.get_scorer(metric[0])
                    continue
                if len(metric) == 1:
                    kw_args = {}
                else:
                    kw_args = copy.deepcopy(metric[1])
                    if 'needs_custom_kw_args' in kw_args:
                        if self.custom_scorer:
                            raise ValueError("Only one custom metric can be set with 'needs_custom_kw_args'.")
                        del kw_args['needs_custom_kw_args']
                        self.custom_scorer = sklearn.metrics.make_scorer(metric[0], **kw_args)
                        scorers[name] = self.custom_scorer_shell
                        continue
                scorers[name] = sklearn.metrics.make_scorer(metric[0], **kw_args)
            else:
                scorers[name] = sklearn.metrics.get_scorer(name)
        return scorers

    def get_refit(self, p):
        # main_score_name come from 'gs__refit'
        # check self.main_score_name in 'metrics' and 'gs__metrics'
        if self.p['gs__refit']:
            refit = self.p['gs__refit']

            if refit not in p['gs__metrics']:
                p['gs__metrics'].append(refit)
                self.logger.warning(f"Warning: grid search 'refit' metric '{refit}' should be present"
                                    f" in 'gs__metrics', added.")
        elif self.p['gs__scoring']:
            refit = self.p['gs__scoring'][0]
            self.logger.warning(f"Warning: gs refit metric not set,"
                                f"zero position from 'gs__metrics' used: {refit}.")
        else:
            # we guaranteed scoring not empty in metrics_to_scorers.
            assert False, "gs__metrics should be set"
            # [deprecated]
            # self.logger.warning(f"Warning: neither 'gs_refit', nor 'gs__metric' set:"
            #                     f"estimator built-in score method is used.")
            # refit = list(self.scorers.keys())[0]
        return refit

    def get_hp_grid(self, hp_grid):
        # if estimator_type == 'classifier':
            # [deprecaated] excessive
            # if self.p['th__strategy'] == 0:
            #     _ = self.p['gs__hp_grid'].pop('estimate__apply_threshold__threshold', None)

            # [deprecated] rename to estimate__predict_proba__classifier
            #else:
            #    # add __clf__ to estimator hps names to pass in PredictTransformer
            #    for name, vals in list(self.p['gs__hp_grid'].items()):
            #        if 'estimate__classifier' in name:
            #            lis = name.split('__')
            #            lis.insert(-1, 'clf')
            #            new_name = '__'.join(lis)
            #            self.p['gs__hp_grid'][new_name] = self.p['gs__hp_grid'].pop(name)

        # hp_grid, remove non unique
        remove_keys = set()
        for key, val in hp_grid['gs__hp_grid'].items():
            # TODO: case with iterator, generator, range
            if hasattr(val, '__iter__'):
                # np.unique use built-in sort => not applicable for objects (dict, list, transformers)
                # pd.unique use built-in set => not work with unhashable types (list of dict, list of list)
                # transform to str, drop repeated and back to dict
                if hasattr(val, '__len__'):
                    if len(val) == 0:
                        remove_keys.add(key)
                    else:
                        if isinstance(val[0], dict):
                            # not prevent repetition
                            # [deprecated] [{}] => [];  [{},{'a':7}] => [{'a': nan}, {'a': 7.0}]
                            # hp_grid['gs__hp_grid'][key] = pd.DataFrame(val).drop_duplicates().to_dict('r')
                            pass
                        else:
                            hp_grid['gs__hp_grid'][key] = pd.unique(val)
        for key in remove_keys:
            del hp_grid['gs__hp_grid'][key]
        return hp_grid

    def custom_scorer_shell(self, estimator, x, y):
        """Read custom_kw_args from current pipeline, pass to scorer.

        Note: in gs self object copy, we can dynamically get param only from estimator,
        """
        try:
            if estimator.steps[0][0] == 'pass_custom':
                if estimator.steps[0][1].kw_args:
                    self.custom_scorer._kwargs.update(estimator.steps[0][1].kw_args)
        except AttributeError:
            # 'ThresholdClassifier' object has no attribute 'steps'
            self.custom_scorer._kwargs.update(self.default_custom_kw_args)

        return self.custom_scorer(estimator, x, y)

    def get_scoring(self, refit):
        """Set gs target score for different strategies."""
        th_range = None
        if self.p['pipeline__type'] == 'classifier':
            if self.p['th__strategy'] == 0:
                scoring = self.scorers
            elif self.p['th__strategy'] == 1:
                if 'estimate__apply_threshold__threshold' in self.p['gs__hp_grid']:
                    th_range = self.p['gs__hp_grid'].pop('estimate__apply_threshold__threshold')
                    self.logger.warning('Warning: brutforce threshold experimental strategy 1.1')
                else:
                    self.logger.warning('Warning: brutforce threshold experimental strategy 1.2')
                self.logger.warning("Warning: add 'roc_auc' to grid search scorers.")
                refit = 'roc_auc'
                scoring = {**self.scorers, 'roc_auc': sklearn.metrics.get_scorer('roc_auc'), }
            elif self.p['th__strategy'] == 2:
                scoring = self.scorers
                if 'estimate__apply_threshold__threshold' in self.p['gs__hp_grid']:
                    self.logger.warning('Warning: brutforce threshold experimental strategy 2.1')
                else:
                    th_range, _, _ = self.calc_th_range()
                    self.p['gs__hp_grid'].update({'estimate__apply_threshold__threshold': th_range})
                    self.logger.warning('Warning: brutforce threshold experimental strategy 2.2')
            elif self.p['th__strategy'] == 3:
                if 'estimate__apply_threshold__threshold' in self.p['gs__hp_grid']:
                    th_range = self.p['gs__hp_grid'].pop('estimate__apply_threshold__threshold')
                    self.logger.warning('Warning: brutforce threshold experimental strategy 3.1')
                else:
                    self.logger.warning('Warning: brutforce threshold experimental strategy 3.2')
                if not isinstance(refit, str):
                    raise ValueError("Error: for strategy 3 'refit' should be of type 'str'")
                refit = f'experimental_{refit}'
                scoring = {**self.scorers, refit: self.scorer_strategy_3}
                self.logger.warning(f"Warning: add {refit} to grid search scorers.")
            else:
                raise MyException("th__strategy should be 0-3")
        else:
            # regression
            scoring = self.scorers

        return scoring, th_range, refit

    def get_n_iter(self):
        """Set gs number of runs"""
        # calculate from hps ranges if user 'gs__runs' is not given
        if self.p['gs__runs'] is None:
            try:
                n_iter = np.prod([len(i) if isinstance(i, list) else i.shape[0]
                                  for i in self.p['gs__hp_grid'].values()])
            except AttributeError as e:
                self.logger.critical("Error: distribution for hyperparameter grid is used,"
                                     " specify 'gs__runs' in params.")
                raise ValueError("distribution for hyperparameter grid is used, specify 'gs__runs' in params.")
        else:
            n_iter = self.p['gs__runs']
        return n_iter

    def get_pre_dispatch(self):
        """Set gs parallel jobs

        If n_jobs was set to a value higher than one, the data is copied for each parameter setting.
        Using pre_dispatch you can set how many pre-dispatched jobs you want to spawn.
        The memory is copied only pre_dispatch many times. A reasonable value for pre_dispatch is 2 * n_jobs.
        n_jobs can be -1, mean spawn all

        """
        # deprecated
        # if self.p['gs__pre_dispatch'] == 'minimal':
        #     pre_dispatch = max(1, self.p['gs__n_jobs']) if self.p['gs__n_jobs'] else 1
        pre_dispatch = self.p['gs__pre_dispatch']
        return pre_dispatch

    def runs_compliance(self, runs, runs_th_, best_index):
        """"Combine GS results to csv dump."""
        # runs.csv compliance
        # add param
        default_th = self.estimator.get_params()['estimate__apply_threshold__threshold']
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

    def gui_compliance(self, best_th_, th_range):
        """"[deprecated] Combine GS results to use gui."""
        pass

    def distribution_compliance(self, res, hp_grid):
        for name, vals in hp_grid.items():
            # check if distribution in hp_grid
            if not hasattr(type(vals), '__iter__'):
                # there are masked array in res
                hp_grid[name] = pd.unique(np.ma.getdata(res[f'param_{name}']))

    # @time_profiler
    def dump_runs(self, runs, best_run_index):
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
        self.logger.info("\u25CF \u25B6 DUMP RUNS")
        # get full params for each run
        nums = len(runs['params'])
        lis = list(range(nums))
        est_clone = sklearn.clone(self.estimator)  # not clone attached data, only params
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
        # df2 = pd.DataFrame([self.estimator.get_params()])

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

        # dump to disk in run dir
        dirpath = '{}/results/runs'.format(self.project_path)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        filepath = '{}/{}_runs.csv'.format(dirpath, int(time.time()))
        with open(filepath, 'a', newline='') as f:
            df.to_csv(f, mode='a', header=f.tell() == 0, index=False, line_terminator='\n')
        self.logger.log(25, f"Save run(s) results to file:\n    {filepath}")
        self.logger.log(25, f"Best run id:\n    {run_id_list[best_run_index]}")
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

    def find_modifiers(self, hp_grid):
        """Get names of hp_grid params setted with range."""
        self.logger.info('hp grid:\n    {}'.format(jsbeautifier.beautify(str(hp_grid))))
        # find varied hp
        modifiers = []
        for key, val in hp_grid.items():
            if isinstance(val, list):
                size = len(val)
            else:
                size = val.shape[0]
            if size > 1:
                modifiers.append(key)
        return modifiers

    def gs_print(self, res, modifiers):
        """nice print"""
        param_modifiers = set('param_'+i for i in modifiers)
        # outputs
        runs_avg = {'mean_fit_time': res.cv_results_['mean_fit_time'].mean(),
                    'mean_score_time': res.cv_results_['mean_score_time'].mean()}
        df = pd.DataFrame(res.cv_results_)[[key for key in res.cv_results_ if key in param_modifiers
                                            or 'mean_train' in key or 'mean_test' in key]]
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            # self.logger.debug('{}'.format(df.head()))
            self.logger.info('{}'.format(tabulate.tabulate(df, headers='keys', tablefmt='psql')))
        # Alternative: df.to_string()

        self.logger.info('GridSearch best index:\n    {}'.format(res.best_index_))
        self.logger.info('GridSearch time:\n    {}'.format(runs_avg))
        self.logger.log(25, 'CV best modifiers:\n'
                            '    {}'.format(jsbeautifier.beautify(str({key: res.best_params_[key]
                                                                       for key in modifiers
                                                                       if key in res.best_params_}))))
        self.logger.info('CV best configuration:\n'
                         '    {}'.format(jsbeautifier.beautify(str(res.best_params_))))
        self.logger.info('CV best mean test score:\n'
                         '    {}'.format(res.__dict__.get('best_score_', 'n/a')))  # not exist if refit callable
        self.logger.info('Errors:\n'
                         '    {}'.format(self.np_error_stat))
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

    # =============================================== threshold calcs ==================================================
    def calc_th_range(self, th_range=None):
        """ Сalculate th range from OOF roc_curve.

        Used in th__strategy (1)(2)(3.2).

        Args:
            th_range (array-like, optional(default=None)): if None, will be calculated from roc_curve.

        Returns:
            th_range
            predict_proba
            y_true (np.ndarray): true labels for self.y_train.

        Note:
            For classification task it is possible to tune classification threshold ``th_`` on CV.
            For details see `Concepts <./Concepts.html#classification-threshold>`__.
            Mlshell support multiple strategy for ``th_`` tuning.

                (0) Don't use ``th_`` (common case).

                    * Not all classificator provide predict_proba (SVM).
                    * We can use f1, logloss.
                    * | If necessary you can dynamically pass params in custom scorer function to tune them in CV,
                      | use 'pass_custom__kw_args' step in hp_grid.

                (1) First GS best hps with CV, then GS best ``th_`` (common case).

                    * For GS hps by default used auc-roc as score.
                    * For GS ``th_`` main score.

                (2) Use additional step in pipeline (metaestimator) to GS ``th_`` in predefined range (experimental).

                    * Tune ``th_`` on a par with other hps.
                    * ``th_`` range should be unknown in advance:

                        (2.1) set in arbitrary in hp_grid

                        (2.2) take typical values from ROC curve OOF

                (3) While GS best hps with CV, select best ``th_`` for each fold in separately (experimental).

                    * For current hp combination maximize tpr/(tpr+fpr) on each fold by ``th_``.
                    * | Although there will different best ``th_`` on folds,
                      | the generalizing ability of classifier might be better.
                    * Then select single overall best ``th_`` on GS with main score.


        TODO:
            | add support TimeSeriesSplit
            | add plot
            |    https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py
            | strategy 4: GS set here

        """
        y_pred_proba, _, y_true = self.cross_val_predict(
            self.estimator, self.x_train, y=self.y_train, groups=None,
            cv=self.cv(), fit_params=self.p['pipeline__fit_params'], method='predict_proba')
        if th_range is None:
            best_th_, best_ind, q, fpr, tpr, th_range = self.brut_th_(y_true, y_pred_proba)
            coarse_th_range, coarse_index = self.coarse_th_range(best_th_, th_range)
            if self.p['th__plot_flag']:
                self.th_plot(y_true, y_pred_proba, best_th_, q, tpr, fpr, th_range, coarse_th_range, coarse_index)
        else:
            coarse_th_range = th_range
        return coarse_th_range, y_pred_proba, y_true

    def th_plot(self, y_true, y_pred_proba, best_th_, q, tpr, fpr, th_, coarse_th_, coarse_index):
        # TODO: built_in roc curve plotter
        #    https: // scikit - learn.org / stable / modules / generated / sklearn.metrics.RocCurveDisplay.html  # sklearn.metrics.RocCurveDisplay
        fig, axs = plt.subplots(nrows=2, ncols=1)
        fig.set_size_inches(10, 10)
        # roc_curve
        roc_auc = sklearn.metrics.roc_auc_score(y_true, y_pred_proba[:, self.pos_label_ind])
        axs[0].plot(fpr, tpr, 'darkorange', label=f"ROC curve (area = {roc_auc:.3f})")
        axs[0].scatter(fpr[coarse_index], tpr[coarse_index], c='b', marker="o")
        axs[0].plot([0, 1], [0, 1], color='navy', linestyle='--')
        axs[0].set_xlabel('False Positive Rate')
        axs[0].set_ylabel('True Positive Rate')
        axs[0].set_title(f"Receiver operating characteristic (label '{self.pos_label}')")
        axs[0].legend(loc="lower right")
        # tpr/(tpr+fpr)
        axs[1].plot(th_, q, 'green')
        axs[1].vlines(best_th_, np.min(q), np.max(q))
        axs[1].vlines(coarse_th_, np.min(q), np.max(q), colors='b', linestyles=':')
        axs[1].set_xlim([0.0, 1.0])
        axs[1].set_xlabel('Threshold')
        axs[1].set_ylabel('TPR/(TPR+FPR)')
        axs[1].set_title('Selected th values near maximum')
        # plt.plot(th_, fpr, 'red')
        plt.show()

    def cross_val_predict(self, *args, **kwargs):
        """Function to make bind OOF prediction/predict_proba.

        Args:
            args
            kwargs

        Returns:
            folds_predict_proba (2d np.ndarray): OOF probability predictions [n_test_samples x n_classes].
            folds_test_index (1d np.ndarray): test indices for OOF subset (reseted, not raw).
            y_true (1d np.ndarray): test for OOF subset (for Kfold whole dataset).

        TODO:
            in some fold could be not all classes, need to check.
        """
        # dev check for custom OOF
        debug = False
        estimator = args[0]
        x = args[1]
        y = kwargs['y']
        cv = kwargs['cv']
        temp_pp = None
        temp_ind = None
        try:
            folds_predict_proba = sklearn.model_selection.cross_val_predict(*args, **kwargs)
            folds_test_index = np.arange(0, folds_predict_proba.shape[0])
            if debug:
                temp_pp = folds_predict_proba
                temp_ind = folds_test_index
                raise ValueError('debug')
        except ValueError as e:
            # custom OOF
            # for TimeSplitter no prediction at first fold
            # self.logger.warning('Warning: {}'.format(e))
            folds_predict_proba = []  # list(range(self.cv_n_splits))
            folds_test_index = []  # list(range(self.cv_n_splits))
            # th_ = [[2, 1. / self.n_classes] for i in self.classes_]  # init list for th_ for every class
            ind = 0
            for fold_train_index, fold_test_index in cv.split(x):
                # stackingestimator__sample_weight=train_weights[fold_train_subindex]
                if hasattr(x, 'loc'):
                    estimator.fit(x.loc[x.index[fold_train_index]],
                                  y.loc[y.index[fold_train_index]],
                                  **self.p['pipeline__fit_params'])
                    # in order of self.estimator.classes_
                    fold_predict_proba = estimator.predict_proba(x.loc[x.index[fold_test_index]])
                else:
                    estimator.fit(x[fold_train_index], y[fold_train_index], **self.p['pipeline__fit_params'])
                    # in order of self.estimator.classes_
                    fold_predict_proba = estimator.predict_proba(x[fold_test_index])
                # merge th_ for class
                # metrics.roc_curve(y[fold_test_index], y_test_prob, pos_label=self.pos_label)
                # th_[self.pos_label].extend(fold_th_)
                folds_test_index.extend(fold_test_index)
                folds_predict_proba.extend(fold_predict_proba)
                ind += 1
            folds_predict_proba = np.array(folds_predict_proba)
            folds_test_index = np.array(folds_test_index)
            # delete duplicates
            # for i in range(self.n_classes):
            #    th_[i] = sorted(list(set(th_[i])), reverse=True)
        if debug:
            assert np.array_equal(temp_pp, folds_predict_proba)
            assert np.array_equal(temp_ind, folds_test_index)

        y_true = y.values[folds_test_index] if hasattr(y, 'loc') else y[folds_test_index]
        return folds_predict_proba, folds_test_index, y_true

    def coarse_th_range(self, best_th_, th_):
        """Get most possible th range.

        Note:
            linear sample from [best/100; 2*best] with limits [np.min(th), 1]
            th descending
            th_range ascending
        """
        th_range_desire = np.linspace(max(best_th_ / 100, np.min(th_)), min(best_th_ * 2, 1), self.p['th__samples'])
        # find index of nearest from th_reverse
        index_rev = np.searchsorted(th_[::-1], th_range_desire, side='left')  # a[i-1] < v <= a[i]
        index = len(th_) - index_rev - 1
        th_range = np.clip(th_[index], a_min=None, a_max=1)
        return th_range, index

    def brut_th_(self, y_true, y_pred_proba):
        """ Measure th value that maximize tpr/(fpr+tpr).

        Note:
            for th gs will be used values near best th.

        TODO:
            it is possible to bruforce based on self.metric
            early-stopping if q decrease

        """
        fpr, tpr, th_ = sklearn.metrics.roc_curve(
            y_true, y_pred_proba[:, self.pos_label_ind],
            pos_label=self.pos_label, drop_intermediate=True)
        # th_ sorted descending
        # fpr sorted ascending
        # tpr sorted ascending
        # q go through max
        q = np_divide(tpr, fpr+tpr)  # tpr/(fpr+tpr)
        best_ind = np.argmax(q)
        best_th_ = th_[best_ind]
        # [deprecated] faster go from left
        # use reverse view, need last occurrence
        # best_th_ = th_[::-1][np.argmax(q[::-1])]
        return best_th_, best_ind, q, fpr, tpr, th_

    def prob_to_pred(self, y_pred_proba, th_):
        """Fix threshold on predict_proba"""
        y_pred = np.where(y_pred_proba[:, self.pos_label_ind] > th_, [self.pos_label], [self.neg_label])
        return y_pred

    # =============================================== cv ===============================================================
    def cv(self, n_splits=None):
        """Method to generate samples for cv

        Args:
            n_splits (int): number of splits

        Returns:
            object which have split method yielding train/test splits indices
        """
        if n_splits is not None:
            self.p['gs__splitter'].n_splits = n_splits

        return self.p['gs__splitter']

    # =============================================== scorers ==========================================================
    def scorer_strategy_1(self, estimator, x, y_true):
        """Calculate score strategy (1) for classification

        Note:
            Predict probabilities.
            Calculate roc_auc.

        """
        y_pred_proba = estimator.predict_proba(x)
        score = sklearn.metrics.roc_auc_score(y_true, y_pred_proba[:, self.pos_label_ind])
        return score

    def scorer_strategy_3(self, estimator, x, y_true):
        """Calculate score strategy (2) for classification

        Note:
            if ``score`` scorer work with y_predictions:
                Predict probabilities.
                Brutforce th from roc_curve range.
                Score main metric after fix best th.

        """
        scorer = self.scorers[self.refit]
        metric = scorer._score_func
        if isinstance(scorer, sklearn.metrics._scorer._PredictScorer):
            y_pred_proba = estimator.predict_proba(x)
            best_th_ = self.brut_th_(y_true, y_pred_proba)[0]
            y_pred = self.prob_to_pred(y_pred_proba, best_th_)
            score = metric(y_true, y_pred)
        elif isinstance(scorer, sklearn.metrics._scorer._ProbaScorer):
            y_pred_proba = estimator.predict_proba(x)
            score = metric(y_true, y_pred_proba[:, self.pos_label_ind])
        elif isinstance(scorer, sklearn.metrics._scorer._ThresholdScorer):
            y_pred_decision = estimator.decision_funcrion(x)
            score = metric(y_true, y_pred_decision)
        else:
            raise MyException("Can`t understand scorer type")

        return score