"""ML workflow class.
TODO: in fit, optimize maybe want use only part of dataset => allow kwargs in split
    also get_X() get_y()  mayve change to data, target attributes with generation.

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
TODO:
    think about replace get_x, get_y with cycled generators

TEST:
    multioutput, better under targets column.
        not sure if th_resolve would work
        maybe need to move in dataset or sklearn utils (predict_proba to pos_labels_pp, prob_to_pred)
        this is not dataset quality, it-is sklearn quality => utills.sklearn dir
        code hould not repete!
    multiclass, should work for all except th_strategy.
    not dataframe format for data.
    pass_csutom, n_jobs save?
    result reproducible
    extract categor_ind_name/numeric_ind_name
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


class Workflow(mlshell.Producer):
    """Class for ml workflow."""
    # TODO:
    # _required_parameters = [,]

    def __init__(self, project_path='', logger=None, endpoint_id='default_workflow',
                 datasets=None, pipelines=None, metrics=None, params=None):
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
        self.logger = logger if logger else logging.Logger(__class__.__name__)
        super().__init__(self.project_path, self.logger)

        self.endpoint_id = endpoint_id
        self.logger.info("\u25CF INITITALIZE WORKFLOW")
        self.datasets = datasets if datasets else {}
        self.pipelines = pipelines if pipelines else {}
        self.metrics = metrics if metrics else {}
        self.params = params if params else {}

        self.check_results_size(project_path)

        # depends on pipeline_id and dataset_id
        self._runs = {}

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

        self.logger.info('Used params:\n    {}'.format(jsbeautifier.beautify(str(self.params))))

        # [not full before init()]
        # self.logger.info('Workflow metods:\n    {}'.format(jsbeautifier.beautify(str(self.__dict__))))

        # hash of hp_params
        self.np_error_stat = {}
        np.seterrcall(self.np_error_callback)

        # fullfill in self.unify_data()
        self.classes_ = None
        self.n_classes = None
        self.neg_label = None
        self.pos_labels = None
        self.pos_labels_ind = None
        self.categoric_ind_name = None
        self.numeric_ind_name = None
        self.data_hash = None
        # [deprected] make separate function call
        # self.unify_data(data)

        # for pass custom only (thread-unsafe)
        self.custom_scorer = {'n/a': None}
        self.cache_custom_kwargs = {'n/a':{}}
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

    def __hash__(self):
        # TODO: hash of it`s self
        return md5(str(self.params).encode('utf-8')).hexdigest()

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

    # TODO: NOT SURE YET
    def check_data_format(self, data, params):
        """check data format"""

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
    def add_dataset(self, dataset, dataset_id):
        """Add dataset to workflow internal storage.

        Args:
            dataset (): .
            dataset_id (str): .

        """
        # [alternative]
        # dataset = self.data_check(dataset)
        self.datasets.update({dataset_id: dataset})
        return

    def pop_dataset(self, dataset_ids):
        """Pop data from wotkflow data storage.

        Args:
            dataset_ids (str, iterable): ids to pop.
        Return:
            popped data dict.
        """
        if isinstance(dataset_ids, str):
            dataset_ids = [dataset_ids]
        return {dataset_id: self.datasets.pop(dataset_id, None)
                for dataset_id in dataset_ids}

    def add_pipeline(self, pipeline, pipeline_id):
        """Add pipeline to workflow internal storage.

        Args:
            pipeline (): .
            pipeline_id (str): .

        """
        self.pipelines.update({pipeline_id: pipeline})
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
    def _check_arg(self, arg, func=None):
        """Check if argument is id or object."""
        if isinstance(arg, str):
            return arg
        else:
            assert False, 'Argument should be str'
            # TODO[beta]: пока оставлю так,пусть дата и пайплайн всегда через config задаются, потом можно расширить
            # fit() принимает pipeline_id вместо  pipeline, но read_conf резолвит pipeline
            # вообще у даты и пайплайн особый статус, но с другими параметрами должна быть синхронность
            # Лучше так: можно и id  и напрямую пайплайн, дату, это будет логично.
            # тогда не будет отличатся от других. Только там внутри есть хранилища зависимые от айдишников.
            # надо дефолтный айди тогда создавать!
            # pipeline should contain pipeline.pipeine

            # Generate arbitrary.
            # id = str(int(time.time()))
            # Add to storage under id.
            # func(arg, id)
            # return id

    @checker
    # @memory_profiler
    def fit(self, pipeline, dataset, **kwargs):
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
        pipeline_id = self._check_arg(pipeline)
        dataset_id = self._check_arg(dataset)

        dataset = self.datasets[dataset_id]
        pipeline = self.pipelines[pipeline_id]
        # resolve and set hps
        pipeline = self._set_hps(pipeline, dataset, **kwargs)
        # optional
        self._print_steps(pipeline)
        # [deprecated] excessive
        # if kwargs.get('debug', False):
        #     self.debug_pipeline_(pipeline, dataset)

        train, test = dataset.split()
        # [deprecated] now more abstract
        # x_train, y_train, _, _ = dataset.split()

        self.logger.info("\u25CF FIT PIPELINE")
        # [deprecated] separate fit and optimize
        # if not kwargs.get('gs',{}).get('flag', False):
        pipeline.fit(train.get_x(), train.get_y(), **kwargs.get('fit_params', {}))
        # [deprecated] dump not needed, no score evaluation
        # best_run_index = 0
        # runs = {'params': [self.estimator.get_params(),]}
        self.pipelines[pipeline_id] = pipeline

    def optimize(self, pipeline, dataset, optimizer, validator, **kwargs):
        pipeline_id = self._check_arg(pipeline)
        dataset_id = self._check_arg(dataset)

        dataset = self.datasets[dataset_id]
        pipeline = self.pipelines[pipeline_id]
        # For pass_custom.
        self.current_pipeline_id = pipeline_id
        # Resolve and set hps.
        pipeline = self._set_hps(pipeline, dataset, **kwargs)
        # Resolve hp_grid.
        hp_grid = kwargs['gs_params'].pop('hp_grid', {})
        if hp_grid:
            # [deprecated] kwargs['gs_params']['hp_grid'] =
            hp_grid = self._resolve_hps(hp_grid, pipeline, dataset, **kwargs)

        # Resolve scoring.
        scoring = kwargs['gs_params'].pop('scoring', {})
        if scoring:
            kwargs['gs_params']['scoring'] = validator(logger=self.logger).resolve_scoring(scoring, self.metrics,
                                                                                           **pipeline.pipeline.get_params())
        train, test = dataset.split()

        self.logger.info("\u25CF \u25B6 OPTIMIZE HYPERPARAMETERS")
        optimizer = optimizer(pipeline.pipeline, hp_grid, **kwargs.get('gs_params', {}))
        optimizer.fit(train.get_x(), train.get_y(), **kwargs.get('fit_params', {}))

        # Results logs/dump to disk in run dir.
        dirpath = '{}/results/runs'.format(self.project_path)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        filepath = '{}/{}_runs.csv'.format(dirpath, int(time.time()))
        optimizer.dump_runs(self.logger, filepath)

        self._runs[(pipeline_id, dataset_id)] = optimizer.update_best(self._runs.get((pipeline_id, dataset_id), {}))
        # [deprecated] for one pipeline could be different optimizer`s interface
        # pipeline.update_params(optimizer)
        if 'best_estimator_' in self._runs[(pipeline_id, dataset_id)]:
            self.pipelines[pipeline_id] = self._runs[(pipeline_id, dataset_id)].get('best_estimator_')
        # [deprecated] not informative
        # else:
        #     self.logger.warning("Warning: optimizer.update_best don`t contain 'best_estimator_':\n"
        #                         "    optimizer results will not be used for pipeline.")

    def _set_hps(self, pipeline, dataset, **kwargs):
        hps = pipeline.pipeline.get_params()
        # [deprecated] currently pipeline change inplace
        # hps.update(pipeline.get('best_params_', {}))
        hps.update(self._get_zero_position(kwargs.get('hp', {})))
        hps = self._resolve_hps(hps, pipeline, dataset, **kwargs)
        pipeline.pipeline.set_params(**hps)
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

    def _resolve_hps(self, hps, pipeline, dataset, **kwargs):
        for hp_name, val in hps.items():
            if val == 'auto':
                # hp
                hps[hp_name] = pipeline.resolve(hp_name, dataset, **kwargs)
            elif val == ['auto']:
                # hp_grid
                hps[hp_name] = [].extend(pipeline.resolve(hp_name, dataset, **kwargs))
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
    # def _is_data_hp(self, val):
    #     return isinstance(val, str) and val.startswith('data__')

    def _print_steps(self, pipeline):
        # nice print of pipeline
        params = pipeline.pipeline.get_params()
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
    def validate(self, pipeline_id, dataset_id, validator, **kwargs):
        """Predict and score on validation set."""
        self.logger.info("\u25CF VALIDATE ON HOLDOUT")
        dataset = self.datasets[dataset_id]
        pipeline = self.pipelines[pipeline_id]
        train, test = dataset.split()

        validator = validator(logger=self.logger)
        metrics = validator.resolve_metric(kwargs.get('metric', []), self.metrics)
        validator.via_metrics(metrics, pipeline, train, test, **kwargs)
        # [deprecated] not all metrics can be converted to scorers
        # validator.via_scorers(self.metrics_to_scorers(self.metrics, self.metrics),
        # pipeline, train, test)
        return

    # =============================================== dump ==========================================================
    def dump(self, pipeline_id, dirpath=None):
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
        if not dirpath:
            dirpath = '{}/results/models'.format(self.project_path)

        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        fit_dataset_id = getattr(pipeline, 'dataset_id', None)
        best_score = str(self._runs.get((pipeline_id, fit_dataset_id), {}).get('best_score_', '')).lower()
        filepath = f"{dirpath}/{self.endpoint_id}_{pipeline_id}_{fit_dataset_id}_" \
                   f"{best_score}_{hash(self)}_{hash(pipeline)}_{hash(self.datasets[fit_dataset_id])}_dump.model"
        if not os.path.exists(filepath):
            # prevent double dumping
            pipeline.dump(filepath)
            self.logger.log(25, 'Save fitted model to file:\n  {}'.format(filepath))
        else:
            self.logger.warning('Warnning: skip dump: model file already exists\n    {}\n'.format(filepath))

        # alternative:
        # with open(file, 'wb') as f:
        #     pickle.dump(self.estimator, f)
        return filepath

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
    def predict(self, pipeline_id, dataset_id, dirpath=None, template=None):
        """Predict on new dataset.

        Args:
            data (pd.DataFrame): data ready for workflow unification.
            raw_names (dict): {'index': 'index_names', 'targets': 'target_names', 'feature_names'}.
            estimator (sklearn-like estimator, optional (default=None)): fitted estimator,
                if None use from workflow object.

        """
        self.logger.info("\u25CF PREDICT ON TEST")

        pipeline = self.pipelines[pipeline_id]
        dataset = self.datasets[dataset_id]
        train, test = dataset.split()
        assert train == test
        x = test.get_x()

        # [deprecated]
        # data_df, _, _ = self.unify_data(data)
        # x_df = data_df.drop(['targets'], axis=1)  # was used for compatibility with unifier

        y_pred = pipeline.predict(x)

        # dump to disk in predictions dir
        if not template:
            template = test.get_y()
        if not dirpath:
            dirpath = '{}/results/models'.format(self.project_path)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        fit_dataset_id = pipeline.__dict__.get('dataset_id', None)
        best_score = str(self._runs.get((pipeline_id, fit_dataset_id), {}).get('best_score_', '')).lower()
        filepath = f"{dirpath}/{self.endpoint_id}_{pipeline_id}_{fit_dataset_id}_" \
                   f"{best_score}_{hash(self)}_{hash(pipeline)}_{hash(self.datasets[fit_dataset_id])}_" \
                   f"{dataset_id}_{hash(self.datasets[dataset_id])}_predictions"

        dataset.dump(filepath, y_pred, template)
        self.logger.log(25, f"Save predictions dataset {dataset_id} to file:\n    {filepath}")

    # =============================================== gui param ========================================================
    def gui(self, pipeline_id, dataset_id, hp_grid, optimizer_id, cls,  **kwargs):
        self.logger.info("\u25CF GUI")

        pipeline = self.pipelines[pipeline_id]
        dataset = self.datasets[dataset_id]

        # we need only hp_grid flat:
        # either hp here in args
        # either combine tested hps for all optimizers if hp = {}
        runs = self._runs.get((pipeline_id, dataset_id), {})
        gui = cls(pipeline, dataset, runs, **kwargs)
        threading.Thread(target=gui.plot(), args=(), daemon=True).start()
        return


if __name__ == '__main__':
    pass
