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

TEST:
    multioutput, better under targets column.
    multiclass, should work for all except th_strategy.
    not dataframe format for data.
"""


import mlshell.custom
import mlshell.default
from mlshell.libs import *
from mlshell.callbacks import dic_flatter, json_keys2int


class Workflow(object):
    """Class for ml workflow."""

    def __init__(self, project_path, logger=None, params=None):
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

        # hash of hp_params
        self.np_error_stat = {}
        np.seterrcall(self.np_error_callback)

        # fullfill in self.unify_data()
        self.classes_ = None
        self.n_classes = None
        self.neg_label = None
        self.pos_label = None
        self.pos_label_ind = None
        self.data = {}
        self.categoric_ind_name = None
        self.numeric_ind_name = None
        self.data_hash = None
        # [deprected] make separate function call
        # self.unify_data(data)

        self.custom_scorer = None
        self.default_custom_kw_args = {}
        # fit
        self.refit = None
        self.scorers = None
        # fullfill in self.create_pipeline()
        self.pipeline = {}
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

    def _workflow_hash(self):
        # TODO: hash of it`s self
        return md5(str(self.p).encode('utf-8')).hexdigest()

    def check_data_format(self, data, params):
        """check data format"""
        # TODO: move to pipeline check
        if not isinstance(data, pd.DataFrame):
            raise TypeError("input data should be pandas.DataFrame object")

        # TODO: move to data check
        if 'targets' not in data.columns:
            raise KeyError("input dataframe should contain 'targets' column, set zero values columns if absent")
        if not all(['feature_' in column for column in data.columns if 'targets' not in column]):
            raise KeyError("all name of dataframe features columns should start with 'feature_'")

        # TODO: workflow level
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
        data = self._data_check(data)
        self.data.update(data)
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
        return {data_id: self.data.pop(data_id, None)
                for data_id in data_ids}

    def add_pipeline(self, pipeline):
        """Add data to workflow pipelines storage.

        Args:
            pipeline(dict): {'pipeline_id': val,}.

        """
        self.pipeline.update(pipeline)
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
        return {pipe_id: self.pipeline.pop(pipe_id, None)
                for pipe_id in pipe_ids}

    # =============================================== pipeline =========================================================
    # TODO: this is analog of sklearn estimator check
    @ check_hash()
    def debug_pipeline_(self):
        """Fit pipeline on small subset for debug"""
        x, y = self.tonumpy(self.data_df[:min(1000, len(self.data_df))])
        fitted = self.estimator.fit(x, y, **self.p['pipeline__fit_params'])
        self.recursive_logger(fitted.steps)

    def recursive_logger(self, steps, level=0):
        """"Recursive log of params for pipeline steps

            Args:
                steps (list): steps on current level
                level (int): level of recursion

        """
        indent = 3
        for step in steps:
            ob_name = step[0]
            ob = step[1]
            self.logger.info('{0}{1}\n{0}{2}'.format('   ' * level, ob_name, ob))
            if hasattr(ob, '__dict__'):
                for attr_name in ob.__dict__:
                    attr = getattr(ob, attr_name)
                    self.logger.info('{0}{1}\n{0}   {2}'.format('   ' * (level + indent), attr_name, attr))
                    if isinstance(attr, list) and (attr_name == 'steps' or attr_name == 'transformers'):
                        self.recursive_logger(attr, level + 1)
                        # [deprecated] specific print
                        # for i in range(1, len(pipeline_)):
                        #     steps = pipeline_[:i]
                        #     last_step = steps[-1][0]
                        #     est = pipeline.Pipeline(steps)
                        #
                        #      if last_step == 'encode_categ':
                        #          temp = est.fit_transform(x).categories_
                        #          for i, vals in enumerate(temp):
                        #              glob_ind = list(self.categoric_ind_name.keys())[i]  # python > 3.0
                        #              self.logger.debug('{}{}\n'.format(self.categoric_ind_name[glob_ind][0],
                        #                                                   self.categoric_ind_name[glob_ind][1][vals]))
                        #      elif last_step =='impute':

    # =============================================== move out ============================================================
    ## pipeline class
    def is_classifier_(self, pipeline):
        return sklearn.base.is_classifier(pipeline)

    def is_regressor_(self, pipeline):
        return sklearn.base.is_regressor(pipeline)

    def dump_(self, pipeline, file):
        joblib.dump(pipeline, file)
        return

    def pipeline_hash_(self, pipeline):
        return 0

    def ckeck_data_format_(self, pipeline, data):
        # call everywhere when move to pipeline
        return

    ## data_class
    def get_from_data_(self, data, key):
        # TODO:
        #      merge with get_classes
        #      data.get(key) inherent from dict
        res = key
        if key == 'raw_names':
            pass
        elif key == 'categoric_ind_name':
            # self.extract_ind_name(data)
            pass
        elif key == 'numeric_ind_name':
            # self.extract_ind_name(data)
            pass
        elif key == 'classes':
            pass
        elif key == 'hash':
            res = pd.util.hash_pandas_object(data.get('df')).sum()
        return res

    def dump_predict_(self, filepath, y_pred, data):
        raw_names = self.get_from_data_(data, 'raw_names')
        raw_index_names = raw_names['index']
        raw_targets_names = raw_names['targets']

        y_pred_df = pd.DataFrame(index=data['df'].index.values,
                                 data={raw_targets_names[0]: y_pred}).rename_axis(raw_index_names)

        with open(f"{filepath}.csv", 'w', newline='') as f:
            y_pred_df.to_csv(f, mode='w', header=True, index=True, sep=',', line_terminator='\n')  # only LF
        return

    def get_classes_(self, data, is_classifier, pos_label=None):
        # TODO: multi-output target
        # TODO: remove ['targets'] everywhere or 'targets__origin_targets'
        #   then i can skip raw_names/indices. base_plot better specify in gui!
        # Pipeline also no need excessive attributes, but we need methods somehow.

        if is_classifier:
            classes = np.unique(data['targets'])
            # [deprecated] easy to get from classes
            # n_classes = classes_.shape[0]

            if not pos_label:
                pos_label = classes[-1]  # self.p['th__pos_label']
                pos_label_ind = -1
            else:
                pos_label_ind = np.where(classes == pos_label)[0][0]
            # [deprecated] multiclass
            # neg_label = classes[0]
            self.logger.info(f"Label {pos_label} identified as positive np.unique(targets)[-1]:\n"
                             f"    for classifiers provided predict_proba:"
                             f" if P(pos_label)>threshold, prediction=pos_label on sample.")
        else:
            classes = None
            pos_label = None
            pos_label_ind = None
        return classes, pos_label, pos_label_ind

    def split_(self, data):
        # TODO: better move to dataproperties, because could be non dataframe
        # check for:
        #    fit
        #    validate

        df = data['df']
        train_index = data.get('train_index', None)
        test_index = data.get('test_index', None)
        if not train_index and not test_index:
            train_index = test_index = df.index

        columns = df.columns
        # deconcatenate without copy, better dataframe over numpy (provide index)
        train = df.loc[train_index]
        test = df.loc[test_index]
        x_train = train[[name for name in columns if 'feature' in name]]
        y_train = train['targets']
        x_test = test[[name for name in columns if 'feature' in name]]
        y_test = test['targets']
        return (x_train, y_train), (x_test, y_test)

    def extract_ind_name(self, data):
        categoric_ind_name = {}
        numeric_ind_name = {}
        for ind, column_name in enumerate(data):
            if 'targets' in column_name:
                continue
            if '_categor_' in column_name:
                # loose categories names
                categoric_ind_name[ind - 1] = (column_name, np.unique(data[column_name]))
            else:
                numeric_ind_name[ind - 1] = (column_name,)
        return data, categoric_ind_name, numeric_ind_name

    # =============================================== gridsearch =======================================================
    @check_hash
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

        data = self.data[data_id]
        pipeline = self.pipeline[pipeline_id]
        pipeline = self._set_hps(pipeline, data, kwargs)

        # optional
        self._print_steps(pipeline)
        if kwargs.get('debug', False):
            self.debug_pipeline_(pipeline, data)

        train, test = self.split_(data)
        # [deprecated] now more abstract
        # x_train, y_train, _, _ = data.split()

        self.logger.info("\u25CF FIT PIPELINE")
        # [deprecated] separate fit and optimize
        # if not kwargs.get('gs',{}).get('flag', False):
        pipeline.fit(*train, **kwargs.get('fit_params', {}))
        # [deprecated] dump not needed, no score evaluation
        # best_run_index = 0
        # runs = {'params': [self.estimator.get_params(),]}

    def optimize(self, pipeline_id, data_id, cls, hp_grid, **kwargs):
        data = self.data[data_id]
        pipeline = self.pipeline[pipeline_id]

        # need only for resolviong data-related hps
        hp_grid = kwargs['gs_params'].pop('hp_grid', {})
        if hp_grid:
            kwargs['gs_params']['hp_grid'] = self._resolve_hps(hp_grid,
                                                               data,
                                                               kwargs)

        # TODO: move out
        # optional
        self._print_steps(pipeline)
        if kwargs.get('debug', False):
            self.debug_pipeline_(pipeline, data)

        train, test = self.split_(data)

        self.logger.info("\u25CF \u25B6 GRID SEARCH HYPERPARAMETERS")
        optimizer = cls(pipeline, hp_grid, **kwargs.get('gs_params', {}))
        optimizer.fit(*train, **kwargs.get('fit_params', {}))

        # TODO:
        runs, best_run_index = self.optimize()
        # dump results in csv
        self.dump_runs(runs, best_run_index)

    def _set_hps(self, pipeline, data, kwargs):
        hps = pipeline.get_params().update(self._get_zero_position(kwargs))
        hps = self._resolve_hps(hps, data, kwargs)
        pipeline.set_params(**hps)
        return pipeline

    def _get_zero_position(self, kwargs):
        """
        Note:
            In case of generator/iterator change in hp_grid will be irreversible.

        """
        # get zero position params from hp
        hps = {}
        for name, vals in kwargs.get(['hp'], {}).items():
            # check if not distribution in hp
            if hasattr(type(vals), '__iter__'):
                # container type
                iterator = iter(vals)
                hps.update(**{name: iterator.__next__()})
        return hps

    def _resolve_hps(self, hps, data, kwargs):
        for hp_name in hps:
            # step_name = step[0]
            # step_hp = {key: p[key] for key in p.keys() if step_name + '__' in key}
            val = hps[hp_name]
            if self._is_data_hp(val):
                key = val.split('__')[-1]
                hps[hp_name] = self.get_from_data_(data, key)
            elif isinstance(val, dict):
                # dict case
                for k,v in val.items():
                    if self._is_data_hp(v):
                        key = v.split('__')[-1]
                        val[k] = self.get_from_data_(data, key)
            elif hasattr(type(val), '__iter__') and\
                    hasattr(type(val), '__getitem__'):
                # sequence case
                for k, v in enumerate(val):
                    if self._is_data_hp(v):
                        key = v.split('__')[-1]
                        val[k] = self.get_from_data_(data, key)
        return hps

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

    # =============================================== validate =========================================================
    # @memory_profiler
    def validate(self, pipeline_id, data_id, **kwargs):
        """Predict and score on validation set."""
        self.logger.info("\u25CF VALIDATE ON HOLDOUT")
        data = self.data[data_id]
        pipeline = self.pipeline[pipeline_id]
        train, test = self.split_(data)

        classes, pos_label, pos_label_ind = \
            self.get_classes_(data,
                              self.is_classifier_(pipeline),
                              kwargs.get('pos_label', None))

        self._via_metrics(kwargs.get('metric', []), pipeline,
                          train, test, pos_label_ind, classes)
        # [deprecated] not all metrics can be converted to scorers
        # self._via_scorers(self.metrics_to_scorers(self.p['metrics'], self.p['metrics']),
        # pipeline, train, test, pos_label_ind)
        return

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

        for name, metric in metrics.items():
            if metric[1].get('needs_proba', False):
                if not hasattr(pipeline, 'predict_proba'):
                    self.logger.warning(f"Warning: pipeline object has no method 'predict_proba':\n"
                                        "    ignore metric '{name}'")
                    continue
                # [...,i] equal to [:,i]/[:,:,i]/.. (for multi-output target)
                y_pred_train = pipeline.predict_proba(x_train)[..., pos_label_ind]
                y_pred_test = pipeline.predict_proba(x_test)[..., pos_label_ind]
            elif metric[1].get('needs_threshold', False):
                if not hasattr(pipeline, 'decision_function'):
                    self.logger.warning(f"Warning: pipeline object has no method 'predict_proba':\n"
                                        "    ignore metric '{name}'")
                    continue
                y_pred_train = pipeline.decision_function(x_train)
                y_pred_test = pipeline.decision_function(x_test)
            else:
                y_pred_train = pipeline.predict(x_train)
                y_pred_test = pipeline.predict(x_test)

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
            self._score_prette_print(metric, score_train, score_test, classes)
        return

    def _score_prette_print(self, metric, score_train, score_test, classes):
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
            score_train = scorer(pipeline, *train)
            # result score on test
            score_test = scorer(pipeline, *test)
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
        pipeline = self.pipeline[pipeline_id]
        # dump to disk in models dir
        dirpath = '{}/results/models'.format(self.project_path)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        file = f"{dirpath}/{self.p_hash}_{self.data_hash}_dump.model"
        if not os.path.exists(file):
            # prevent double dumping
            self.dump_(pipeline, file)
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
    def predict(self, pipeline_id, data_id):
        """Predict on new data.

        Args:
            data (pd.DataFrame): data ready for workflow unification.
            raw_names (dict): {'index': 'index_names', 'targets': 'target_names', 'feature_names'}.
            estimator (sklearn-like estimator, optional (default=None)): fitted estimator,
                if None use from workflow object.

        """
        self.logger.info("\u25CF PREDICT ON TEST")

        pipeline = self.pipeline[pipeline_id]
        data = self.data[data_id]
        train, test = self.split_(data)
        assert train == test
        x, y = test

        # [deprecated]
        # data_df, _, _ = self.unify_data(data)
        # x_df = data_df.drop(['targets'], axis=1)  # was used for compatibility with unifier

        y_pred = pipeline.predict(x)

        # hash of data
        workflow_hash = self._workflow_hash()
        pipeline_hash = self.pipeline_hash_(pipeline)
        data_hash = self.get_from_data_(data, 'hash')

        # dump to disk in predictions dir
        dirpath = '{}/results/models'.format(self.project_path)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        filepath = f"{dirpath}/{workflow_hash}_{pipeline_hash}_{data_hash}_predictions"

        self.dump_predict_(filepath, y_pred, data)
        self.logger.log(25, "Save predictions for new data to file:\n    {}".format(filepath))

    # =============================================== gui param ========================================================
    def gui(self, pipeline_id, data_id, cls,  **kwargs):
        self.logger.info("\u25CF GUI")

        pipeline = self.pipeline[pipeline_id]
        data = self.data[data_id]

        gui = cls(pipeline, data, best_params_, hp_grid ,**kwargs)
        threading.Thread(target=gui.plot(), args=(,), daemon=True).start()
        return

    # TODO: move into GUI
    def gen_gui_params(self):
        """Prepare params for visualization."""
        self.logger.info("\u25CF PREPARE GUI PARAMS")
        # rearrange nested hp params
        hp_grid_flat = {}
        for key, val in self.p['gs__hp_grid'].items():
            if key not in self.modifiers:  # only if multiple values
                continue
            if isinstance(val[0], dict):
                # functiontransformer kw_args compliance (pass_custom)
                # ('pass_custom__kw_args','param_a')
                dic = {tuple([key, key_]): np.zeros(len(val), dtype=type(val[0]), order='C')
                       for key_ in val[0].keys()}
                # [deprecated] problem in inverse transform in gui (can`t understand if needed)
                # 'pass_custom__kw_args__param_a'
                # dic = {'__'.join([key, key_]): np.zeros(len(val), dtype=np.float64, order='C')
                #        for key_ in val[0].keys()}
                for i, item in enumerate(val):
                    for key_ in dic:
                        dic[key_][i] = item[key_[1]]
                        # dic[key_][i] = item[key_.split('__')[-1]]
                hp_grid_flat.update(dic)
            else:
                hp_grid_flat[key] = self.to_numpy(val)

        # not necessary
        best_params_flat = {}
        dic_flatter(self.best_params_, best_params_flat, key_transform=tuple, val_transform=self.to_numpy)

        self.gui_params = {
            'pipeline__type': self.p['pipeline__type'],
            'data': self.data_df,
            'train_index': self.train_index,
            'test_index': self.test_index,
            'estimator': self.estimator,
            'gs__hp_grid': self.p['gs__hp_grid'],       # {'param':range,}
            'best_params_': self.best_params_,          # {'param':value,}
            'hp_grid_flat': hp_grid_flat,               # {'param':range,}
            'best_params_flat': best_params_flat,       # {'param':value,}
            'metric': self.metric,
        }

    def to_numpy(self, val):
        """Hp param to numpy.

        Note:
            object transform to np object
            float force to np.float64
            https://docs.scipy.org/doc/numpy/reference/arrays.scalars.html

        """
        if isinstance(val, list):
            typ = type(val[0])
            val = np.array(val, order='C', dtype=np.float64 if typ is float else typ)
        #    if isinstance(val[0], (str, bool, int, np.number)):
        #        val = np.array(val, order='C', dtype=type(val[0]))
        #    else:
        #        try:
        #            # try cast to double
        #            val = np.array(val, order='C', dtype=np.double)
        #        except Exception as e:
        #            # cast to string, otherwise would save as object, would be problem with sort further
        #            val = np.array([str(i) for i in val], order='C')
        # elif not isinstance(val, (str, bool, int, np.number)):
        #     val = str(val)

            # [deprecated] not work for non-built-in objects
            # if isinstance(val[0], str):
            #     val = np.array(val, order='C')
            # elif isinstance(val[0], bool):
            #     val = np.array(val, order='C', dtype=np.bool)
            # elif isinstance(val[0], int):
            #     val = np.array(val, order='C', dtype=int)
            # else:
            #     val = np.array(val, order='C', dtype=np.double)
        return val

    def custom_scorer(self, estimator, x, y_true, greater_is_better=True, needs_proba=False, needs_threshold=False):
        """Custom scorer.

        Args:
            estimator: fitted estimator.
            x (dataframe, np.ndarray): features test.
            y_true (dataframe, np.ndarray): true targets test.

        Returns:
            score value

        Note:
            Alternative use built-in make_scorer
                scorer = metrics.make_scorer(metrics.accuracy_score, greater_is_better=True)
                score = scorer(estimator, x, y)

        """
        if isinstance(x, pd.DataFrame):
            index = y_true.index
            x = x.values
            y_true = y_true.values

        if needs_proba and needs_threshold:
            raise ValueError("Set either needs_proba or needs_threshold to True,"
                             " but not both.")

        if needs_proba:
            y_pred_proba = estimator.predict_proba(x)
            score = self.metric(y_true, y_pred_proba[:, self.pos_label_ind], y_pred_type='proba')
        elif needs_threshold:
            y_pred_decision = estimator.decision_funcrion(x)
            score = self.metric(y_true, y_pred_decision, y_pred_type='decision')
        else:
            y_pred = estimator.predict(x)
            score = self.metric(y_true, y_pred, y_pred_type='targets')

        if greater_is_better:
            return score
        else:
            return -score

    def metric(self, y_true, y_pred, y_pred_type='targets', meta=False):
        """Evaluate custom metric.

        Args:
            y_true (np.ndarray): true targets.
            y_pred (np.ndarray): predicted targets/pos_label probabilities/decision function.
            meta (bool): if True calculate metadata for visualization.
            y_pred_type: 'targets'/'proba'/'decision'

        Returns:
            score (float): metric score
            meta (dict): cumulative score in dynamic; TP,FP,FN in points for classification.

        """
        if self.p['pipeline__type'] == 'classifier':
            return self.metric_classifier(y_true, y_pred, y_pred_type, meta)
        else:
            return self.metric_regressor(y_true, y_pred, meta)

    def metric_classifier(self, y_true, y_pred, y_pred_type, meta=False):
        """Evaluate classification metric.

        Detailed meta for external use cases.

        Args:
            y_true (np.ndarray): true targets.
            y_pred (np.ndarray): predicted targets/pos_label probabilities/decision function.
            meta (bool): if True calculate metadata for visualization, support only `targets` in y_pred.
            y_pred_type: 'targets'/'proba'/'decision'

        Returns:
            score (float): metric score
            meta (dict): cumulative score in dynamic; TP,FP,FN in points for classification.

        Note:
            Be carefull with y_pred type, there is no check in sklearn.
            meta support only `targets` in y_pred.

        TODO:
            add y_pred_type argument (sklearn not support)
            need to pass threshold from estimator
            add roc_curve on plot

        """
        # score
        scorer_kwargs = self.p['metrics'][self.main_score_name][1] if len(self.p['metrics'][self.main_score_name]) > 1 else {}
        if scorer_kwargs.get('needs_proba') or scorer_kwargs.get('needs_threshold'):
            # [deprecated] confusing
            # self.logger.warning("Warning:  gui classification metric don`t suport predict_proba "
            #                     "or decision_function based metrics\n    "
            #                     "score set to None")
            score = None  # sklearn.metrics.f1_score(y_true, y_pred, pos_label=self.pos_label)
        else:
            score = self.p['metrics'][self.main_score_name][0](y_true, y_pred)

        if not meta:
            return score

        if y_pred_type is not 'targets':
            raise ValueError("metric_classification meta don`t support non-target input for predictions")

        precision_score = sklearn.metrics.precision_score(y_true, y_pred)
        # metrics in dynamic
        length = y_true.shape[0]
        tp = 0
        fp = 0
        fn = 0
        tp_fn = 0
        precision_vector = np.zeros(length, dtype=np.float64)
        tp_vector = np.zeros(length, dtype=np.bool)
        fp_vector = np.zeros(length, dtype=np.bool)
        fn_vector = np.zeros(length, dtype=np.bool)
        for i in range(length):
            if y_true[i] == 1 and y_pred[i] == 1:
                tp += 1
                tp_vector[i] = True
            elif y_true[i] == 0 and y_pred[i] == 1:
                fp += 1
                fp_vector[i] = True
            elif y_true[i] == 1 and y_pred[i] == 0:
                fn += 1
                fn_vector[i] = True
            if y_true[i] == 1:
                tp_fn += 1
            precision_vector[i] = tp / (fp + tp) if tp + fp != 0 else 0

        if precision_score != precision_vector[-1]:
            assert False, 'MyError: score_check False'

        meta = {'score': precision_vector, 'TP': tp_vector, 'FP': fp_vector, 'FN': fn_vector}

        return score, meta

    def metric_regressor(self, y_true, y_pred, meta=False):
        """Evaluate regression metric.

        Detailed meta for external use cases.

        Args:
            y_true (np.ndarray): true targets.
            y_pred (np.ndarray): predicted targets. need for strategy=1 (auc-score)
            meta (bool): if True calculate metadata for visualization

        Returns:
            score (float): metric score
            meta (dict): cumulative score, mae,mse in dynamic; resid in points.

        """
        # check for Inf prediction (in case of overfitting), limit it
        if np.isinf(y_pred).sum():
            np.nan_to_num(y_pred, copy=False)

        # score
        score = self.p['metrics'][self.main_score_name][0](y_true, y_pred)
        if not meta:
            return score

        # end value
        r2_score = sklearn.metrics.r2_score(y_true, y_pred)
        mae_loss = sklearn.metrics.mean_absolute_error(y_true, y_pred)
        mse_loss = sklearn.metrics.mean_squared_error(y_true, y_pred)

        # metrics in dynamic
        length = y_true.shape[0]
        mae_vector = np.zeros(length, dtype=np.float64)
        mse_vector = np.zeros(length, dtype=np.float64)
        resid_vector = np.zeros(length, dtype=np.float64)
        r2_vector = np.zeros(length, dtype=np.float64)
        mae = 0
        mse = 0
        mean = 0
        # tss, rss, r2 (don`t need initialization)
        for n in range(length):
            # cumul
            mae = (mae*n + abs(y_true[n]-y_pred[n]))/(n+1)
            mse = (mse*n + (y_true[n]-y_pred[n])**2)/(n+1)
            mean = (mean*n + y_true[n])/(n+1)
            rss = mse*(n+1)
            tss = np.sum((y_true[:n+1]-mean)**2)   # tss + (y_true[n]-mean)**2
            r2 = 1 - rss/tss if tss != 0 else 0
            r2_vector[n] = r2
            mae_vector[n] = mae
            mse_vector[n] = mse
            # in points
            resid_vector[n] = y_pred[n]-y_true[n]

        for score_, score_vector_ in [(r2_score, r2_vector), (mae_loss, mae_vector), (mse_loss, mse_vector)]:
            if not cmath.isclose(score_, score_vector_[-1], rel_tol=1e-8, abs_tol=0):  # slight difference
                assert False, 'MyError: score_check False'

        meta = {'score': r2_vector, 'MAE': mae_vector, 'MSE': mse_vector, 'RES': resid_vector}

        return score, meta


if __name__ == '__main__':
    pass
