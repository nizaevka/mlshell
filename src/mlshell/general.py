"""ML workflow class.
TODO: All API check
    https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
    multiclass, multioutput
    https://scikit-learn.org/stable/modules/multiclass.html#multiclass
TODO: There were error in old with load if new workflow, function addresses changes.
    now i use special {'indices':'data__categoric_ind_names'}
TODO: add clusterer

TODO:
    better numpy style, easy  to replace
    do google then change
    https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html
    https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator
TODO:
    resolve scoring more beautiful

TEST:
    multioutput, better under targets column.
    multiclass, should work for all except th_strategy.
    not dataframe format for data.
    pass_csutom, n_jobs save?
"""


import mlshell.custom
import mlshell.default
from mlshell.libs import *
from mlshell.callbacks import dic_flatter, json_keys2int


def checker(function_to_decorate, options=None):
    """Decorator to check alteration in hash(self.data_df) after call method"""
    # TODO: multioptional
    #     add self._np_error_stat flush and check at the end!
    #     update hash check
    # https://stackoverflow.com/questions/10294014/python-decorator-best-practice-using-a-class-vs-a-function/10300995
    if options is None:
        options = []
    def wrapper(*args, **kwargs):
        self = args[0]
        # before = pd.util.hash_pandas_object(self.data_df).sum()
        function_to_decorate(*args, **kwargs)
        # after = pd.util.hash_pandas_object(self.data_df).sum()
        # assert before == after, ""
        self.logger.info('Errors:\n'
                         '    {}'.format(self.np_error_stat))
    return wrapper


class Workflow(object):
    """Class for ml workflow."""

    def __init__(self, project_path, logger=None, params=None, datasets=None, pipelines=None):
        """Initialize workflow object

        Args:
            project_path (str): path to project dir.
            logger (logging.Logger): logger object.
            params (dict): user workflow configuration params.

        Attributes:
            self.project_path (str): path to project dir.
            self.logger (:obj:'logger'): logger object.
            self.p (dict): user workflow configuration params, for skipped one used default.
            self.p_hash (str): md5 hash of params.
            self.data_df (pd.DataFrame): data before split.
            self.np_error_stat (dict): storage for np.error raises.
            self.classes_(np.ndarray): class labels in classification.
            self.n_classes (int): number of target classes in classification.
            self.neg_label (target type): negative label.

        Note:
            dataframe should have columns={'targets', 'feature_<name>', 'feature_categor_<name>'}

                * 'feature_categor_<name>': any dtype (include binary).

                    order is not important.

                * 'feature_<name>': any numeric dtype (should support float(val), np.issubdtype(type(val), np.number))

                    order is important.

                * 'targets': any dtype

                    for classification `targets` should be binary, ordinalencoded.
                    | positive label should be > others in np.unique(y) sort.

        """
        self.project_path = project_path
        if logger is None:
            self.logger = logging.Logger('Workflow')
        else:
            self.logger = logger
        self.logger.info("\u25CF INITITALIZE WORKFLOW")
        self.datasets = datasets if datasets else {}
        self.pipelines = pipelines if pipelines else {}

        self.check_results_size(project_path)


        # [deprecated]
        # merge in read conf
        # check_params_vals in fit
        # # use default if skipped in params
        # temp = copy.deepcopy(mlshell.default.DEFAULT_PARAMS)
        # if params is not None:
        #     self.check_params_keys(temp, params)
        #     temp.update(params)
        # self.check_params_vals(temp)
        # self.p = temp

        self.p = params
        self.logger.info('Used params:\n    {}'.format(jsbeautifier.beautify(str(self.p))))

        # [not full before init()]
        # self.logger.info('Workflow metods:\n    {}'.format(jsbeautifier.beautify(str(self.__dict__))))

        # hash of hp_params
        self.np_error_stat = {}
        np.seterrcall(self.np_error_callback)

        # fullfill in self.unify_data()
        self.classes_ = None
        self.n_classes = None
        self.neg_label = None
        self.pos_label = None
        self.pos_label_ind = None
        self.categoric_ind_name = None
        self.numeric_ind_name = None
        self.data_hash = None
        # [deprected] make separate function call
        # self.unify_data(data)

        # for pass custom only (thread-unsafe)
        self.custom_scorer = {'n/a': None}
        self.cache_custom_kw_args = {'n/a':{}}
        self.current_pipeline_id = 'n/a'

        # fit
        self.refit = None
        self.scorers = None
        # fullfill in self.split()
        self.train_index = None
        self.test_index = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        # fulfill in self.fit()
        self.best_params_ = {}
        self.modifiers = []
        # fulfill in self.gen_gui_params()
        self.gui_params = {}
        # TODO: Decide use self.metrics or self.p['metrics'] ?
        #       We extent sklearn metrics namespace, in optimize/validate pass only names

    def __hash__(self):
        # TODO: hash of it`s self
        return md5(str(self.p).encode('utf-8')).hexdigest()

    def check_results_size(self, project_path):
        root_directory = pathlib.Path(f"{project_path}/results")
        size = sum(f.stat().st_size for f in root_directory.glob('**/*') if f.is_file())
        size_mb = size/(2**30)
        # check if > n Mb
        n = 5
        if size_mb > n:
            self.logger.warning(f"Warning: results/ directory size {size_mb:.2f}Gb more than {n}Gb")

    def np_error_callback(self, *args):
        """Numpy errors handler, count errors by type"""
        if args[0] in self.np_error_stat.keys():
            self.np_error_stat[args[0]] += 1
        else:
            self.np_error_stat[args[0]] = 1

    def check_data_format(self, data, params):
        """check data format"""

        # TODO: NOT SURE YET

        # TODO: move to data check
        if 'targets' not in data.columns:
            raise KeyError("input dataframe should contain 'targets' column, set zero values columns if absent")
        if not all(['feature_' in column for column in data.columns if 'targets' not in column]):
            raise KeyError("all name of dataframe features columns should start with 'feature_'")

        # TODO: pipeline level
        # mayby only with th_strategy != 0
        if params['pipeline__type'] == 'classifier':
            if self.n_classes > 2:
                raise ValueError('Currently only binary classification supported.')

    # =============================================== add/pop ============================================================
    def add_data(self, data):
        """Add data to workflow data storage.

        Args:
            data (dict): {'data_id': val,}.

        """
        # [alternative]
        # data = self.data_check(data)
        self.datasets.update(data)
        return

    def pop_data(self, data_ids):
        """Pop data from wotkflow data storage.

        Args:
            data_ids (str, iterable): ids to pop.
        Return:
            popped data dict.
        """
        if isinstance(data_ids, str):
            data_ids = [data_ids]
        return {data_id: self.datasets.pop(data_id, None)
                for data_id in data_ids}

    def add_pipeline(self, pipeline):
        """Add data to workflow pipelines storage.

        Args:
            pipeline(dict): {'pipeline_id': val,}.

        """
        self.pipelines.update(pipeline)
        return

    def pop_pipeline(self, pipe_ids):
        """Pop pipeline from wotkflow pipeline storage.

        Args:
            pipe_ids (str, iterable): ids to pop.
        Return:
            popped pipelines dict.
        """
        if isinstance(pipe_ids, str):
            pipe_ids = [pipe_ids]
        return {pipe_id: self.pipelines.pop(pipe_id, None)
                for pipe_id in pipe_ids}

    # =============================================== gridsearch =======================================================
    @checker
    # @memory_profiler
    def fit(self, pipeline_id, data_id=None, **kwargs):
        """Tune hp, fit best.
            https://scikit-learn.org/stable/modules/grid_search.html#grid-search

        Args:
            gs_flag (bool): If True tune hp with GridSearch else fit on self.x_train.

        Note:
            RandomizedSearch could duplicate runs (sample with replacement if at least one hp set with distribution)
            The verbosity level:
                * if non zero, progress messages are printed.
                * If more than 10, all iterations are reported.
                * Above 50, the output is sent to stdout.
                 The frequency of the messages increases with the verbosity level.

            If gs_flag is True run grid search else just fit estimator
        """
        data = self.datasets[data_id]
        pipeline = self.pipelines[pipeline_id]
        # resolve and set hps
        pipeline = self._set_hps(pipeline, data, kwargs)
        # optional
        self._print_steps(pipeline)
        # [deprecated] excessive
        # if kwargs.get('debug', False):
        #     self.debug_pipeline_(pipeline, data)

        train, test = data.split()
        # [deprecated] now more abstract
        # x_train, y_train, _, _ = data.split()

        self.logger.info("\u25CF FIT PIPELINE")
        # [deprecated] separate fit and optimize
        # if not kwargs.get('gs',{}).get('flag', False):
        pipeline.fit(train.get_x(), train.get_y(), **kwargs.get('fit_params', {}))
        # [deprecated] dump not needed, no score evaluation
        # best_run_index = 0
        # runs = {'params': [self.estimator.get_params(),]}

    def optimize(self, pipeline_id, data_id, cls, **kwargs):

        data = self.datasets[data_id]
        pipeline = self.pipelines[pipeline_id]
        # For pass_custom.
        self.current_pipeline_id = pipeline_id
        # Resolve and set hps.
        pipeline = self._set_hps(pipeline, data, kwargs)
        # Resolve hp_grid.
        hp_grid = kwargs['gs_params'].pop('hp_grid', {})
        if hp_grid:
            kwargs['gs_params']['hp_grid'] = self._resolve_hps(hp_grid,
                                                               pipeline,
                                                               data,
                                                               kwargs)
        # Resolve scoring.
        # TODO: maybe on read_conf step or inside Optimizer
        scoring = kwargs['gs_params'].pop('scoring', {})
        if scoring:
            kwargs['gs_params']['scoring'] = self._resolve_scoring(scoring, self.p['metric'])

        train, test = data.split()

        self.logger.info("\u25CF \u25B6 OPTIMIZE HYPERPARAMETERS")
        optimizer = cls(pipeline, hp_grid, **kwargs.get('gs_params', {}))
        optimizer.fit(train.get_x(), train.get_y(), **kwargs.get('fit_params', {}))

        # Results logs/dump to disk in run dir.
        dirpath = '{}/results/runs'.format(self.project_path)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        filepath = '{}/{}_runs.csv'.format(dirpath, int(time.time()))
        optimizer.dump(filepath)

        # internall in fit
        # optimizer.print()
        # optimizer.dump_runs()

        # optimizer object should contain:
        # best_params_ (not available if refit is False and multi-metric)
        # best_estimator if refit is not False
        self.pipelines[pipeline_id] = optimizer.__dict__.get('best_estimator_', pipeline)
        self.pipelines[pipeline_id]['best_params_'].updata(optimizer.__dict__.get('best_params_', {}))
        self.cache_custom_kw_args[pipeline_id] = \
            self.pipelines[pipeline_id]['best_params_']\
                .get('pass_custom__kw_args', {})

        return

    def _set_hps(self, pipeline, data, kwargs):
        hps = pipeline.get_params()\
            .update(pipeline.get('best_params_', {}))\
            .update(kwargs.get(['hp'], {}))
        hps = self._resolve_hps(hps, pipeline, data, kwargs)
        hps = self._get_zero_position(hps)
        pipeline.set_params(**hps)
        return pipeline

    def _get_zero_position(self, hps):
        """
        Note:
            In case of generator/iterator change in hp_grid will be irreversible.

        """
        # get zero position params from hp
        zero_hps = {}
        for name, vals in hps.items():
            # check if not distribution in hp
            if hasattr(type(vals), '__iter__'):
                # container type
                iterator = iter(vals)
                zero_hps.update(**{name: iterator.__next__()})
        return zero_hps

    def _resolve_hps(self, hps, pipeline, data, kwargs):
        for hp_name, val in hps.items():
            if val == 'auto':
                # hp
                hps[hp_name] = pipeline.resolve(hp_name, data, kwargs)
            elif val == ['auto']:
                # hp_grid
                hps[hp_name] = [].extend(pipeline.resolve(hp_name, data, kwargs))
        return hps

    # [deprecated] explicit param to resolve in hp_grid
    # def _set_hps(self, pipeline, data, kwargs):
    #     hps = pipeline.get_params().update(self._get_zero_position(kwargs))
    #     hps = self._resolve_hps(hps, data, kwargs)
    #     pipeline.set_params(**hps)
    #     return pipeline

    # def _resolve_hps(self, hps, data, kwargs):
    #     for hp_name in hps:
    #         # step_name = step[0]
    #         # step_hp = {key: p[key] for key in p.keys() if step_name + '__' in key}
    #         val = hps[hp_name]
    #         if self._is_data_hp(val):
    #             key = val.split('__')[-1]
    #             hps[hp_name] = self.get_from_data_(data, key)
    #         elif isinstance(val, dict):
    #             # dict case
    #             for k,v in val.items():
    #                 if self._is_data_hp(v):
    #                     key = v.split('__')[-1]
    #                     val[k] = self.get_from_data_(data, key)
    #         elif hasattr(type(val), '__iter__') and\
    #                 hasattr(type(val), '__getitem__'):
    #             # sequence case
    #             for k, v in enumerate(val):
    #                 if self._is_data_hp(v):
    #                     key = v.split('__')[-1]
    #                     val[k] = self.get_from_data_(data, key)
    #     return hps

    def _is_data_hp(self, val):
        return isinstance(val, str) and val.startswith('data__')

    def _print_steps(self, pipeline):
        # nice print of pipeline
        params = pipeline.get_params()
        self.logger.debug('Pipeline steps:')
        for i, step in enumerate(params['steps']):
            step_name = step[0]
            step_hp = {key: params[key] for key in params.keys() if step_name + '__' in key}
            self.logger.debug('  ({})  {}\n    {}'.format(i, step[0], step[1]))
            self.logger.debug('    hp:\n   {}'.format(jsbeautifier.beautify(str(step_hp))))
        self.logger.debug('+' * 100)
        return

    def _resolve_scoring(self, names, user_metrics):
        """Make scorers from user_metrics.

        Args:
            names (sequence of str): user_metrics names to use in gs.
            user_metrics (dict): {'name': (sklearn metric object, bool greater_is_better), }

        Returns:
            scorers (dict): {'name': sklearn scorer object, }

        Note:
            if 'gs__metric_id' is None, estimator default will be used.

        """
        scorers = {}
        self.custom_scorer[self.current_pipeline_id] = None

        # deprecated
        # if not names:
        #     # need to set explicit, because always need not None 'refit' name
        #     # can`t extract estimator built-in name, so use all from validation user_metrics
        #     names = user_metrics.keys()
        for name in names:
            if name in user_metrics:
                metric = user_metrics[name]
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
                        if self.custom_scorer[self.current_pipeline_id]:
                            raise ValueError("Only one custom metric can be set with 'needs_custom_kw_args'.")
                        del kw_args['needs_custom_kw_args']
                        self.custom_scorer[self.current_pipeline_id] = sklearn.metrics.make_scorer(metric[0], **kw_args)
                        scorers[name] = self._custom_scorer_shell
                        continue
                scorers[name] = sklearn.metrics.make_scorer(metric[0], **kw_args)
            else:
                scorers[name] = sklearn.metrics.get_scorer(name)
        return scorers

    def _custom_scorer_shell(self, estimator, x, y):
        """Read custom_kw_args from current pipeline, pass to scorer.

        Note: in gs self object copy, we can dynamically get param only from estimator.
        """
        try:
            if estimator.steps[0][0] == 'pass_custom':
                if estimator.steps[0][1].kw_args:
                    self.custom_scorer[self.current_pipeline_id]._kwargs.update(estimator.steps[0][1].kw_args)
        except AttributeError:
            # ThresholdClassifier object has no attribute 'steps'
            self.custom_scorer[self.current_pipeline_id]._kwargs.update(self.cache_custom_kw_args[self.current_pipeline_id])

        return self.custom_scorer[self.current_pipeline_id](estimator, x, y)

    # =============================================== validate =========================================================
    # @memory_profiler
    def validate(self, pipeline_id, data_id, **kwargs):
        """Predict and score on validation set."""
        self.logger.info("\u25CF VALIDATE ON HOLDOUT")
        data = self.datasets[data_id]
        pipeline = self.pipelines[pipeline_id]
        train, test = data.split()

        classes, pos_label, pos_label_ind = \
            self.get_classes_(data,
                              pipeline.is_classifier,
                              kwargs.get('pos_label', None))

        metrics = self._resolve_metric(kwargs.get('metric', []), self.p['metric'])

        # TODO: move out to separate replacable class
        self._via_metrics(metrics, pipeline,
                          train, test, pos_label_ind, classes)
        # [deprecated] not all metrics can be converted to scorers
        # self._via_scorers(self.metrics_to_scorers(self.p['metrics'], self.p['metrics']),
        # pipeline, train, test, pos_label_ind)
        return

    def _resolve_metric(self, names, user_metrics):
        res = {}
        for name in names:
            if name in user_metrics:
                res.update({name:user_metrics[name]})
            else:
                # Sklearn built-in, resolve through scorer.
                scorer = sklearn.metrics.get_scorer(name)
                metric = (
                    scorer._score_func,
                    {'greater_is_better': scorer._sign > 0,
                    'needs_proba':
                        isinstance(scorer,
                                   sklearn.metrics._scorer._ProbaScorer),
                    'needs_threshold':
                        isinstance(scorer,
                                   sklearn.metrics._scorer._ThresholdScorer),})
                res.update({name: metric})
        return res

    def _via_metrics(self, metrics, pipeline,
                     train, test, pos_label_ind, classes):
        # via metrics
        #   upside: only one inference
        #   donwside: errpr-prone
        #   strange: xgboost lib contain auto detection y_type logic.
        if not metrics:
            self.logger.warning("Warning: no metrics to evaluate.")
            return
        x_train, y_train = train
        x_test, y_test = test

        # [deprecated] not need anymore
        # if hasattr(pipeline, 'predict_proba'):
        #     th_ = pipeline.get_params().get('estimate__apply_threshold__threshold', 0.5)
        #     y_pred_train = self.prob_to_pred(y_pred_proba_train, th_)
        #     y_pred_test = self.prob_to_pred(y_pred_proba_test, th_)

        # prevent multiple prediction
        temp = {'predict_proba': None,
                'decision_function': None,
                'predict': None}
        for name, metric in metrics.items():
            if metric[1].get('needs_proba', False):
                if not hasattr(pipeline, 'predict_proba'):
                    self.logger.warning(f"Warning: pipeline object has no method 'predict_proba':\n"
                                        "    ignore metric '{name}'")
                    continue
                # [...,i] equal to [:,i]/[:,:,i]/.. (for multi-output target)
                if not temp['predict_proba']:
                    y_pred_train = pipeline.predict_proba(x_train)[..., pos_label_ind]
                    y_pred_test = pipeline.predict_proba(x_test)[..., pos_label_ind]
                    temp['predict_proba'] = (y_pred_train, y_pred_test)
                else:
                    y_pred_train, y_pred_test = temp['predict_proba']
            elif metric[1].get('needs_threshold', False):
                if not hasattr(pipeline, 'decision_function'):
                    self.logger.warning(f"Warning: pipeline object has no method 'predict_proba':\n"
                                        "    ignore metric '{name}'")
                    continue
                if not temp['decision_function']:
                    y_pred_train = pipeline.decision_function(x_train)
                    y_pred_test = pipeline.decision_function(x_test)
                    temp['decision_function'] = (y_pred_train, y_pred_test)
                else:
                    y_pred_train, y_pred_test = temp['decision_function']
            else:
                if not temp['predict']:
                    y_pred_train = pipeline.predict(x_train)
                    y_pred_test = pipeline.predict(x_test)
                    temp['predict'] = (y_pred_train, y_pred_test)
                else:
                    y_pred_train, y_pred_test = temp['predict']

            # skip make_scorer params
            kw_args = {key: metric[1][key] for key in metric[1]
                       if key not in ['greater_is_better', 'needs_proba',
                                      'needs_threshold', 'needs_custom_kw_args']}

            # pass_custom steo inly
            if metric[1].get('needs_custom_kw_args', False):
                if (hasattr(pipeline, 'steps') and
                    pipeline.steps[0][0] == 'pass_custom' and
                    pipeline.steps[0][1].kw_args):
                    kw_args.update(pipeline.steps[0][1].kw_args)

            # result score on Train
            score_train = metric[0](y_train, y_pred_train, **kw_args)
            # result score on test
            score_test = metric[0](y_test, y_pred_test, **kw_args)

            self.logger.log(25, f"{name}:")
            self.logger.log(5, f"{name}:")
            self._score_pretty_print(metric, score_train, score_test, classes)
        return

    def _score_pretty_print(self, metric, score_train, score_test, classes):
        if metric[0].__name__ == 'confusion_matrix':
            labels = metric[1].get('labels', classes)
            score_train = tabulate.tabulate(pd.DataFrame(data=score_train,
                                                         columns=labels,
                                                         index=labels),
                                            headers='keys', tablefmt='psql').replace('\n', '\n    ')
            score_test = tabulate.tabulate(pd.DataFrame(data=score_test,
                                                        columns=labels,
                                                        index=labels),
                                           headers='keys', tablefmt='psql').replace('\n', '\n    ')
        elif metric[0].__name__ == 'classification_report':
            if isinstance(score_train, dict):
                score_train = tabulate.tabulate(pd.DataFrame(score_train),
                                                headers='keys', tablefmt='psql').replace('\n', '\n    ')
                score_test = tabulate.tabulate(pd.DataFrame(score_test),
                                               headers='keys', tablefmt='psql').replace('\n', '\n    ')
            else:
                score_train = score_train.replace('\n', '\n    ')
                score_test = score_test.replace('\n', '\n    ')
        elif isinstance(score_train, np.ndarray):
            # pretty printing numpy arrays
            score_train = np.array2string(score_train, prefix='    ')
            score_test = np.array2string(score_test, prefix='    ')
        self.logger.log(25, f"Train:\n    {score_train}\n"
                            f"Test:\n    {score_test}")
        self.logger.log(5, f"Train:\n    {score_train}\n"
                           f"Test:\n    {score_test}")
        return

    def _via_scorers(self, scorers, pipeline,
                     train, test, pos_label_ind, classes):
        # via scorers
        # upside: sklearn consistent
        # downside:
        #   requires multiple inference
        #   non-score metric not possible (confusion_matrix)

        for name, scorer in scorers.items():
            self.logger.log(25, f"{name}:")
            self.logger.log(5, f"{name}:")
            # result score on Train
            score_train = scorer(pipeline, train.get_x(), train.get_y())
            # result score on test
            score_test = scorer(pipeline, test.get_x(), test.get_y())
            self.logger.log(25, f"Train:\n    {score_train}\n"
                                f"Test:\n    {score_test}")
            self.logger.log(5, f"Train:\n    {score_train}\n"
                               f"Test:\n    {score_test}")
        # non-score metrics
        add_metrics = {name: self.p['metrics'][name] for name in self.p['metrics'] if name not in scorers}
        self._via_metrics(add_metrics, pipeline, train, test, pos_label_ind, classes)

    # =============================================== dump ==========================================================
    def dump(self, pipeline_id):
        """Dump fitted model on disk/string.

        Note:
            pickle can dump on disk/string
                 s = _pickle.dumps(self.estimator)
                 est = pickle.loads(s)
            joblib more efficient on disk
                dump(est, path)
                est = load('filename.joblib')

        """
        self.logger.info("\u25CF DUMP MODEL")
        pipeline = self.pipelines[pipeline_id]
        # dump to disk in models dir
        dirpath = '{}/results/models'.format(self.project_path)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        file = f"{dirpath}/{self.p_hash}_{self.data_hash}_dump.model"
        if not os.path.exists(file):
            # prevent double dumping
            pipeline.dump(file)
            self.logger.log(25, 'Save fitted model to file:\n  {}'.format(file))
        else:
            self.logger.warning('Warnning: skip dump: model file already exists\n    {}\n'.format(file))

        # alternative:
        # with open(file, 'wb') as f:
        #     pickle.dump(self.estimator, f)
        return file

    # =============================================== load ==========================================================
    # [deprecated] there special class to load pipeline and set
    # def load(self, file):
    #     """Load fitted model on disk/string.

    #     Note:
    #         Better use only the same version of sklearn.

    #     """
    #     self.logger.info("\u25CF LOAD MODEL")
    #     pipeline = joblib.load(file)
    #     self.logger.info('Load fitted model from file:\n    {}'.format(file))

    #     # alternative
    #     # with open(f"{self.project_path}/sump.model", 'rb') as f:
    #     #     self.estimator = pickle.load(f)

    # =============================================== predict ==========================================================
    # @memory_profiler
    def predict(self, pipeline_id, data_id, filepath=None, template=None):
        """Predict on new data.

        Args:
            data (pd.DataFrame): data ready for workflow unification.
            raw_names (dict): {'index': 'index_names', 'targets': 'target_names', 'feature_names'}.
            estimator (sklearn-like estimator, optional (default=None)): fitted estimator,
                if None use from workflow object.

        """
        self.logger.info("\u25CF PREDICT ON TEST")

        pipeline = self.pipelines[pipeline_id]
        data = self.datasets[data_id]
        train, test = data.split()
        assert train == test
        x = test.get_x()

        # [deprecated]
        # data_df, _, _ = self.unify_data(data)
        # x_df = data_df.drop(['targets'], axis=1)  # was used for compatibility with unifier

        y_pred = pipeline.predict(x)

        # dump to disk in predictions dir
        if not template:
            template = test.get_y()
        if not filepath:
            dirpath = '{}/results/models'.format(self.project_path)
            if not os.path.exists(dirpath):
                os.makedirs(dirpath)
            filepath = f"{dirpath}/{hash(self)}_{hash(pipeline)}_{hash(data)}_predictions"
        data.dump(filepath, y_pred, template)
        self.logger.log(25, "Save predictions for new data to file:\n    {}".format(filepath))

    # =============================================== gui param ========================================================
    def gui(self, pipeline_id, data_id, hp_grid, optimizer_id, cls,  **kwargs):
        self.logger.info("\u25CF GUI")

        pipeline = self.pipelines[pipeline_id]
        data = self.datasets[data_id]
        optimizer = self.optimizer[optimizer_id]

        # we need only hp_grid flat:
        # either hp here in args
        # either combine tested hps for all optimizers if hp = {}
        gui = cls(pipeline, data, optimizer, hp_grid, **kwargs)
        threading.Thread(target=gui.plot(), args=(), daemon=True).start()
        return


if __name__ == '__main__':
    pass
