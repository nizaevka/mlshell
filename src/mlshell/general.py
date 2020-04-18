"""ML workflow class."""


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
        # use default if skipped in params
        temp = copy.deepcopy(mlshell.default.DEFAULT_PARAMS)
        if params is not None:
            self.check_params_keys(temp, params)
            temp.update(params)
        self.check_params_vals(temp)
        self.p = temp
        self.logger.info('Used params:\n    {}'.format(jsbeautifier.beautify(str(self.p))))

        # hash of hp_params
        self.p_hash = md5(str(self.p).encode('utf-8')).hexdigest()
        self.np_error_stat = {}
        np.seterrcall(self.np_error_callback)

        # fullfill in self.unify_data()
        self.classes_ = None
        self.n_classes = None
        self.neg_label = None
        self.pos_label = None
        self.pos_label_ind = None
        self.data_df = None
        self.categoric_ind_name = None
        self.numeric_ind_name = None
        self.data_hash = None
        # [deprected] make separate function call
        # self.unify_data(data)

        self.custom_scorer = None
        self.default_custom_kw_args = {}
        self.main_score_name = self.p['gs__refit']
        self.scorers = self.metrics_to_scorers(self.p['metrics'], self.p['gs__metrics'])
        # fullfill in self.create_pipeline()
        self.estimator = None
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

    def check_params_keys(self, default_params, params):
        miss_keys = set()
        for key in params:
            if key not in default_params:
                miss_keys.add(key)
                del params[key]
        if miss_keys:
            self.logger.warning(f"Ignore unknown key(s) in conf.py params, check\n    {miss_keys}")

    def check_params_vals(self, params):
        # hp_grid, remove non unique
        # [deprecated] use zero-position, need to clean user error
        # if 'gs__hp_grid' in params:
        remove_keys = set()
        for key, val in params['gs__hp_grid'].items():
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
                            # params['gs__hp_grid'][key] = pd.DataFrame(val).drop_duplicates().to_dict('r')
                            pass
                        else:
                            params['gs__hp_grid'][key] = pd.unique(val)
        for key in remove_keys:
            del params['gs__hp_grid'][key]

        # main_score_name come from 'gs__refit'
        # check self.main_score_name in 'metrics' and 'gs__metrics'
        main_score_name = params['gs__refit']
        if params['gs__flag']:
            if main_score_name not in params['metrics']:
                raise KeyError(f"Warning: gs refit metric '{main_score_name}' should be present"
                               f" in 'metrics'.")

            if main_score_name not in params['gs__metrics']:
                params['gs__metrics'].append(main_score_name)
                self.logger.warning(f"Warning: gs refit metric '{main_score_name}' should be present"
                                    f" in 'gs__metrics', '{main_score_name}' added.")

    def check_data_format(self, data, params):
        """check data format"""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("input data should be pandas.DataFrame object")
        if 'targets' not in data.columns:
            raise KeyError("input dataframe should contain 'targets' column, set zero values columns if absent")
        if not all(['feature_' in column for column in data.columns if 'targets' not in column]):
            raise KeyError("all name of dataframe features columns should start with 'feature_'")
        if params['pipeline__type'] == 'classifier':
            if self.n_classes > 2:
                raise ValueError('only binary classification with pos_label={}'.format(params['th__pos_label']))
            if params['th__pos_label'] != self.classes_[-1]:
                raise ValueError("pos_label={} should be last in np.unique(targets), current={}"
                                 .format(params['th__pos_label'], self.classes_))
        # check that all non-categoric features are numeric type
        # [deprecated] move to self.unify_data (some object-type column could be casted float())

    def np_error_callback(self, *args):
        """Numpy errors handler, count errors by type"""
        if args[0] in self.np_error_stat.keys():
            self.np_error_stat[args[0]] += 1
        else:
            self.np_error_stat[args[0]] = 1

    # =============================================== unify ============================================================
    def set_data(self, data_id, data=None):
        """ Unify dataframe in compliance to workflow class.

        Arg:
            data_id (str):
                identification key for datasets from params['data']
                used as prefix to cache file if `'cache__unifier'` flag is True
            data (:py:class:``pandas.DataFrame``, optional (default=None)):
                object (save original row index after deletes row, need reindex).

        Note:
            If ``use_unifier_cache`` is True:

                * If ``update_unifier_cache`` if False, try to load cache if available (``data`` arg can be skipped).
                * If cache is None or `update_unifier_cache`` is True, run unifier on ``data`` and dump cache after.

            Else: run unifer without cahing results.

        """
        self.logger.info("\u25CF SET DATA")
        if data_id in self.p['data']:
            del_duplicates = self.p['data__del_duplicates']
        else:
            raise KeyError(f"Unknown data_id {data_id}, key should be in params['data'] dictionary.")

        if self.p['cache__unifier'] and not self.p['cache__unifier'] == 'update':
            cache, meta = self.load_instead_unify(prefix=data_id)
            if cache is not None:
                data = cache
                self.data_df = data
                self.categoric_ind_name = meta['categoric']
                self.numeric_ind_name = meta['numeric']
        else:
            cache = None

        if data is None:
            raise ValueError("Set `data` arg in unify_data or turn on 'cache__unifier' in conf.py")

        if self.p['pipeline__type'] == 'classifier':
            self.classes_ = np.unique(data['targets'])
            self.n_classes = self.classes_.shape[0]
            self.pos_label = self.p['th__pos_label']
            self.neg_label = self.classes_[0]
            self.pos_label_ind = np.where(self.classes_ == self.pos_label)[0][0]

        self.check_data_format(data, self.p)

        if cache is None:
            self.check_duplicates(data, del_duplicates)
            self.check_gaps(data)
            if self.p['unify__flag']:
                self.data_df, self.categoric_ind_name, self.numeric_ind_name = self.unify_data(data)
            else:
                self.data_df, self.categoric_ind_name, self.numeric_ind_name = self.extract_ind_name(data)
            self.check_numeric_types(data)
            if self.p['cache__unifier']:
                self.dump_after_unifier(self.data_df, self.categoric_ind_name,
                                        self.numeric_ind_name, prefix=data_id)

        # hash of data before split
        self.data_hash = pd.util.hash_pandas_object(self.data_df).sum()

    def dump_after_unifier(self, data, categoric_ind_name, numeric_ind_name, prefix=''):
        """Dump imtermediate dataframe to disk."""
        cachedir = f"{self.project_path}/results/cache/unifier"
        filepath = f'{cachedir}/{prefix}_after_unifier.csv'
        filepath_meta = f'{cachedir}/{prefix}_after_unifier_meta.json'
        if self.p['cache__unifier']:
            if self.p['cache__unifier'] == 'update':
                if os.path.exists(filepath):
                    os.remove(filepath)
                if os.path.exists(filepath):
                    os.remove(filepath_meta)
                # shutil.rmtree(cachedir, ignore_errors=True)
            if not os.path.exists(cachedir):
                # create temp dir for cache if not exist
                os.makedirs(cachedir)
            # only if cache is None
            self.logger.warning('Warning: update unifier cache file:\n    {}'.format(filepath))
            with open(filepath, 'w', newline='') as f:
                data.to_csv(f, mode='w', header=True, index=True, line_terminator='\n')
            column_ind_name = {'categoric': categoric_ind_name, 'numeric': numeric_ind_name}
            with open(filepath_meta, 'w') as f:
                json.dump(column_ind_name, f)

    def load_instead_unify(self, prefix=''):
        """Load imtermediate dataframe from disk"""
        cachedir = f"{self.project_path}/results/cache/unifier"
        filepath = f'{cachedir}/{prefix}_after_unifier.csv'
        filepath_meta = f'{cachedir}/{prefix}_after_unifier_meta.json'
        if self.p['cache__unifier'] \
                and os.path.exists(filepath) \
                and os.path.exists(filepath_meta) \
                and not self.p['cache__unifier'] == 'update':
            with open(filepath, 'r') as f:
                cache = pd.read_csv(f, sep=",", index_col=0)
            with open(filepath_meta, 'r') as f:
                meta = json.load(f, object_hook=json_keys2int)
            self.logger.warning(f"Warning: use cache file instead unifier:\n    {cachedir}")
            return cache, meta
        return None, None

    def check_duplicates(self, data, del_duplicates):
        # find duplicates rows
        mask = data.duplicated(subset=None, keep='first')  # duplicate rows index
        dupl_n = np.sum(mask)
        if dupl_n:
            self.logger.warning('Warning: {} duplicates rows found,\n    see debug.log for details.'.format(dupl_n))
            # count unique duplicated rows
            rows_count = data[mask].groupby(data.columns.tolist())\
                .size().reset_index().rename(columns={0: 'count'})
            rows_count.sort_values(by=['count'], axis=0, ascending=False, inplace=True)
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                self.logger.debug('Duplicates found\n{}\n'
                                  .format(tabulate.tabulate(rows_count, headers='keys', tablefmt='psql')))

        if del_duplicates:
            # delete duplicates, (not reset index, otherwise problem with base_plot)
            size_before = data.size
            data.drop_duplicates(keep='first', inplace=True)
            # data.reset_index(drop=True, inplace=True) problem with base_plot
            size_after = data.size
            if size_before - size_after != 0:
                self.logger.warning('Warning: delete duplicates rows ({} values)\n'.format(size_before - size_after))

    def check_gaps(self, data):
        # calculate amount of gaps
        gaps_number = data.size - data.count().sum()
        # log
        columns_with_gaps_dic = {}
        if gaps_number > 0:
            for column_name in data:
                column_gaps_namber = data[column_name].size - data[column_name].count()
                if column_gaps_namber > 0:
                    columns_with_gaps_dic[column_name] = column_gaps_namber
            self.logger.warning('Warning: gaps found: {} {:.3f}%,\n'
                                '    see debug.log for details.'.format(gaps_number, gaps_number / data.size))
            self.logger.debug('Gaps per column:\n{}'.format(jsbeautifier.beautify(str(columns_with_gaps_dic))))

        if 'targets' in columns_with_gaps_dic:
            raise MyException("MyError: gaps in targets")
            # delete rows with gaps in targets
            # data.dropna(self, axis=0, how='any', thresh=None, subset=[column_name], inplace=True)

    def check_numeric_types(self, data):
        # check that all non-categoric features are numeric type
        dtypes = data.dtypes
        misstype = []
        for ind, column_name in enumerate(data):
            if '_categor_' not in column_name:
                if not np.issubdtype(dtypes[column_name], np.number):
                    misstype.append(column_name)
        if misstype:
            raise ValueError("Input data non-categoric columns"
                             " should be subtype of np.number, check:\n    {}".format(misstype))

    def unify_data(self, data):
        """ unify input dataframe

        Note:
            * delete duplicates, (not reset index, otherwise problem with base_plot).
            * log info about gaps.
            * unify gaps.

                * if gap in targets => raise MyException
                * if gap in categor => 'unknown'(downcast dtype to str) => ordinalencoder
                * if gap in non-categor => np.nan
            * transform to np.float64 (python float = np.float = np.float64 = C double = np.double(64 bit processor)).
            * define dics for:

                * self.categoric_ind_name => {1:('feat_n', ['cat1', 'cat2'])}
                * self.numeric_ind_name   => {2:('feat_n',)}

        Returns:
            data (pd.DataFrame): unified input dataframe
            categoric_ind_name (dict): {column_index: ('feature_categr__name',['B','A','C']),}
            numeric_ind_name (dict):  {column_index: ('feature__name',),}

        """


        categoric_ind_name = {}
        numeric_ind_name = {}
        for ind, column_name in enumerate(data):
            if 'targets' in column_name:
                continue
            if '_categor_' in column_name:
                # fill gaps with 'unknown'
                # inplace unreliable (could not work without any error)
                # copy!
                data[column_name] = data[column_name].fillna(value='unknown', method=None, axis=None,
                                                             inplace=False, limit=None, downcast=None)
                # copy!
                data[column_name] = data[column_name].astype(str)
                # encode
                encoder = sklearn.preprocessing.OrdinalEncoder(categories='auto')
                data[column_name] = encoder.fit_transform(data[column_name].values.reshape(-1, 1))
                # ('feature_categor__name',['B','A','C'])
                # tolist need to json.dump in cache
                categoric_ind_name[ind-1] = (column_name,
                                             encoder.categories_[0].tolist())
            else:
                # fill gaps with np.nan
                data[column_name].fillna(value=np.nan, method=None, axis=None,
                                         inplace=True, limit=None, downcast=None)
                numeric_ind_name[ind-1] = (column_name,)
        # cast to np.float64 without copy
        # alternative: try .to_numeric
        data = data.astype(np.float64, copy=False, errors='ignore')
        return data, categoric_ind_name, numeric_ind_name

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

    # =============================================== pipeline =========================================================
    @check_hash
    def create_pipeline(self):
        """  Create pipeline

        Note:
            it is possible to use cache
                cache each transformer after calling fit
                avoid double calculation of transformers in GridSearch
                will use cache result if steps and params are the same
            but error-prone (better user-level control)
                https://scikit-learn.org/stable/modules/compose.html#caching-transformers-avoid-repeated-computation
                https://github.com/scikit-learn/scikit-learn/issues/10068
                bad-tested
                will be problem in case of transformer is changed internally
                giant hdd consuming
                time consuming create hash from GB of dat

        """
        self.logger.info("\u25CF CREATE PIPELINE")
        if self.p['cache__pipeline']:
            cachedir = f"{self.project_path}/results/cache/pipeline"
            # delete cache if necessary
            if self.p['cache__pipeline'] == 'update' and os.path.exists(cachedir):
                shutil.rmtree(cachedir, ignore_errors=True)
                self.logger.warning(f'Warning: update pipeline cache:\n    {cachedir}')
            else:
                self.logger.warning(f'Warning: pipeline use cache:\n    {cachedir}')
            if not os.path.exists(cachedir):
                # create temp dir for cache if not exist
                os.makedirs(cachedir)
        else:
            cachedir = None

        # assemble several steps that can be cross-validated together
        pipeline_ = self.pipeline_steps()
        last_step = self.create_last(self.p['pipeline__estimator'], pipeline_)
        self.logger.info(f"Estimator step:\n    {last_step}")
        pipeline_.append(('estimate', last_step))
        self.estimator = sklearn.pipeline.Pipeline(pipeline_, memory=cachedir)

    def create_last(self, estimator, pipeline_):
        """Create last step of pipeline

        Args:
            estimator (sklearn estimator object): to use in last step
            pipeline_ (list of pipeline steps):will use repack 'estimate' for regression

        Returns:
            last_step (pipeline object): last_step

        Note:
            if regression: will use 'estimate' if provided
            if classification: will raise error 'estimate', add custom threshold tuner

        """
        if self.p['pipeline__type'] == 'regressor':
            if pipeline_[-1][0] == 'estimate':
                transformer = pipeline_[-1][1].__dict__['transformer']
                del pipeline_[-1]
            else:
                transformer = None
            last_step = sklearn.compose.TransformedTargetRegressor(regressor=estimator,
                                                                   transformer=transformer, check_inverse=True)
        elif self.p['pipeline__type'] == 'classifier':
            if pipeline_[-1][0] == 'estimate':
                del pipeline_[-1]
            if self.p['th__strategy'] == 0:
                last_step = sklearn.pipeline.Pipeline(steps=[('classifier', estimator)])
                _ = self.p['gs__hp_grid'].pop('estimate__apply_threshold__threshold', None)
            else:
                last_step = sklearn.pipeline.Pipeline(steps=[
                        ('classifier',       mlshell.custom.PredictionTransformer(estimator)),
                        ('apply_threshold',  mlshell.custom.ThresholdClassifier(self.classes_,
                                                                                self.pos_label_ind,
                                                                                self.pos_label,
                                                                                self.neg_label, threshold=0.5)),
                        ])
                # add __clf__ to estimator hps names to pass in PredictTransformer
                for name, vals in list(self.p['gs__hp_grid'].items()):
                    if 'estimate__classifier' in name:
                        lis = name.split('__')
                        lis.insert(-1, 'clf')
                        new_name = '__'.join(lis)
                        self.p['gs__hp_grid'][new_name] = self.p['gs__hp_grid'].pop(name)
        else:
            raise MyException("MyError: unknown estimator type = {}".format(self.p['pipeline__type']))

        if last_step._estimator_type != self.p['pipeline__type']:
            raise MyException('MyError:{}:{}: wrong estimator type'.format(self.__class__.__name__,
                                                                           inspect.stack()[0][3]))
        return last_step

    def pipeline_steps(self):
        """Configure pipeline steps.

        Returns:
            sklearn.pipeline.Pipeline steps.

        Note:
            * | feature/object selections should be independent at every fold,
              | otherwise bias (but ok if totally sure no new data)
            * can`t pass params to fit() in case of TargetTransformation
            * if validate=True in FunctionTransformer, will raise error on "np.nan"
            * limitation of pickling (used for cache and parallel runs on pipeline).

                * define custom function in a module you import, or at least not in a closure.
                * | pickle can`t pickle lambda, cause python pickles by name reference, and a lambda doesn't have a name.
                  | preprocessing.FunctionTransformer(
                  | lambda data: self.subcolumns(data, self.categoric_ind_name),validate=False))

        TODO:
            use dill instead.

        """
        if isinstance(self.p['pipeline__steps'], list):
            steps = self.p['pipeline__steps']
            self.logger.warning('Warning: user-defined pipeline is used instead of default.')
        else:
            if self.p['pipeline__steps'] is None:
                clss = mlshell.default.CreateDefaultPipeline
            else:
                clss = self.p['pipeline__steps']
                self.logger.warning('Warning: user-defined pipeline is used instead of default.')
            steps = clss(self.categoric_ind_name, self.numeric_ind_name, self.p).get_steps()
        return steps

    def debug_pipeline_(self):
        """Fit estimator on whole data for debug"""
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

    # =============================================== split ============================================================
    # @memory_profiler
    def split(self, data=None, data_id=None, **kwargs):
        """Split data on train, test

        data (pandas.DataFrame, optional (default=None)):
            if not None ``data_id`` ignored, read kwargs.
        data_id (str, optional (default='train')):
            | should be known key from params['data`]
            | if None, used default ``data_id`` from params['fit__data_id'] and corresponding kwargs.
        kwargs:
            if data_id is not None, ignore current, use global from params['data__data_id__split'].

        Note:
            input data updated inplace with additional split key.
            if split ``train_size`` set to 1.0, use test=train.
        """
        self.logger.info("\u25CF SPLIT DATA")
        if data:
            data_df = data
        else:
            if not data_id:
                data_id = self.p['fit']['data_id']
            data_df = self.data_df[data_id]
            kwargs = self.p['data'][data_id]['split']

        if (kwargs['train_size'] == 1.0 and kwargs['test_size'] is None
            or kwargs['train_size'] is None and kwargs['test_size'] == 0):
            train = test = data_df
            train_index, test_index = data_df.index
        else:
            train, test, train_index, test_index = sklearn.model_selection.train_test_split(
                data_df, data_df.index.values, **kwargs)
        columns = data_df.columns
        # deconcatenate without copy, better dataframe over numpy (provide index)
        x_train = train[[name for name in columns if 'feature' in name]]
        y_train = train['targets']
        x_test = test[[name for name in columns if 'feature' in name]]
        y_test = test['targets']

        # add to data
        data_df['split'] = {}
        data_df['split']['train_index'] = train_index
        data_df['split']['test_index'] = test_index
        data_df['split']['x_train'] = x_train
        data_df['split']['y_train'] = y_train
        data_df['split']['x_test'] = x_test
        data_df['split']['y_test'] = y_test

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
        scorer = self.scorers[self.main_score_name]
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

    # =============================================== gridsearch =======================================================
    @check_hash
    # @memory_profiler
    def fit(self, data=None, data_id=None, gs_flag=False, pipeline_debug=False, **kwargs):
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
        if data:
            data_df = data
        else:
            if not data_id:
                data_id = self.p['fit']['data_id']
            data_df = self.data_df[data_id]
            kwargs = self.p['gs']
            gs_flag = self.p['fit__gs_flag']
            pipeline_debug = self.p['fit__pipeline_debug']


        self.set_zero_position_hps()
        if pipeline_debug:
            # duplicate nice print from set_zero_position_hps
            self.debug_pipeline_()

        if gs_flag:
            self.logger.info("\u25CF OPTIMIZE PIPELINE")
            runs, best_run_index = self.optimize()
            # dump results in csv
            self.dump_runs(runs, best_run_index)
        else:
            self.logger.info("FIT PIPELINE")
            self.estimator.fit(self.x_train, self.y_train, **self.p['pipeline__fit_params'])
            # dump not needed, no score evaluation
            # best_run_index = 0
            # runs = {'params': [self.estimator.get_params(),]}

    def set_zero_position_hps(self):
        # set zero position params from hp_grid
        for name, vals in self.p['gs__hp_grid'].items():
            # check if distribution in hp_grid
            if hasattr(type(vals), '__iter__'):
                self.estimator.set_params(**{name: vals[0]})

        # nice print of pipeline
        params = self.estimator.get_params()
        self.logger.debug('Pipeline steps:')
        for i, step in enumerate(params['steps']):
            step_name = step[0]
            step_hp = {key: params[key] for key in params.keys() if step_name + '__' in key}
            self.logger.debug('  ({})  {}\n    {}'.format(i, step[0], step[1]))
            self.logger.debug('    hp:\n   {}'.format(jsbeautifier.beautify(str(step_hp))))
        self.logger.debug('+' * 100)

    def optimize(self):
        """Tune hp on train by cv."""
        self.logger.info("\u25CF \u25B6 GRID SEARCH HYPERPARAMETERS")
        # param, fold -> fit(fold_train) -> predict(fold_test) -> score for params
        scoring, th_range = self.get_scoring()
        n_iter = self.get_n_iter()
        pre_dispatch = self.get_pre_dispatch()

        # optimize score
        optimizer = sklearn.model_selection.RandomizedSearchCV(
            self.estimator, self.p['gs__hp_grid'], scoring=scoring, n_iter=n_iter,
            n_jobs=self.p['gs__n_jobs'], pre_dispatch=pre_dispatch,
            refit=self.main_score_name, cv=self.cv(), verbose=self.p['gs__verbose'], error_score=np.nan,
            return_train_score=True).fit(self.x_train, self.y_train, **self.p['pipeline__fit_params'])
        self.estimator = optimizer.best_estimator_
        self.best_params_ = optimizer.best_params_
        best_run_index = optimizer.best_index_
        if 'pass_custom__kw_args' in self.best_params_:
            self.default_custom_kw_args = self.best_params_['pass_custom__kw_args']
        self.distribution_compliance(optimizer.cv_results_, self.p['gs__hp_grid'])
        self.modifiers = self.find_modifiers(self.p['gs__hp_grid'])
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
                n_jobs=1, pre_dispatch=2, refit=self.main_score_name, cv=self.cv(),
                verbose=1, error_score=np.nan, return_train_score=True).fit(predict_proba, y_true,
                                                                            **self.p['pipeline__fit_params'])
            best_th_ = optimizer_th_.best_params_['threshold']
            runs_th_ = copy.deepcopy(optimizer_th_.cv_results_)
            best_run_index = len(runs['params']) + optimizer_th_.best_index_
            self.distribution_compliance(optimizer.cv_results_, self.p['gs__hp_grid'])

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

    def metrics_to_scorers(self, metrics, gs_metrics):
        """Make from scorers from metrics

        Args:
            metrics (dict): {'name': (sklearn metric object, bool greater_is_better), }
            gs_metrics (sequence of str): metrics names to use in gs.

        Returns
            scorers (dict): {'name': sklearn scorer object, }

        """
        scorers = {}
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

    def get_scoring(self):
        """Set gs target score for different strategies."""
        th_range = None
        if self.p['pipeline__type'] == 'classifier':
            if self.p['th__strategy'] == 0:
                scoring = self.scorers
            elif self.p['th__strategy'] == 1:
                scoring = {**self.scorers, self.main_score_name: self.scorer_strategy_1, }
                if 'estimate__apply_threshold__threshold' in self.p['gs__hp_grid']:
                    th_range = self.p['gs__hp_grid'].pop('estimate__apply_threshold__threshold')
                    self.logger.warning('Warning: brutforce threshold experimental strategy 1.1')
                else:
                    self.logger.warning('Warning: brutforce threshold experimental strategy 1.2')
            elif self.p['th__strategy'] == 2:
                scoring = self.scorers
                if 'estimate__apply_threshold__threshold' in self.p['gs__hp_grid']:
                    self.logger.warning('Warning: brutforce threshold experimental strategy 2.1')
                else:
                    th_range, _, _ = self.calc_th_range()
                    self.p['gs__hp_grid'].update({'estimate__apply_threshold__threshold': th_range})
                    self.logger.warning('Warning: brutforce threshold experimental strategy 2.2')
            elif self.p['th__strategy'] == 3:
                scoring = {**self.scorers, self.main_score_name: self.scorer_strategy_3}
                self.logger.warning('Warning: brutforce threshold experimental strategy 3')
                if 'estimate__apply_threshold__threshold' in self.p['gs__hp_grid']:
                    th_range = self.p['gs__hp_grid'].pop('estimate__apply_threshold__threshold')
                    self.logger.warning('Warning: brutforce threshold experimental strategy 3.1')
                else:
                    self.logger.warning('Warning: brutforce threshold experimental strategy 3.2')
            else:
                raise MyException("th__strategy should be 0-3")
        else:
            # regression
            scoring = self.scorers

        return scoring, th_range

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
        self.logger.info('CV best configuration:\n    {}'.format(jsbeautifier.beautify(str(res.best_params_))))
        self.logger.info('CV best mean test score:\n    {}'.format(res.best_score_))
        self.logger.info('Errors:\n    {}'.format(self.np_error_stat))
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

    # =============================================== validate =========================================================
    # @memory_profiler
    def validate(self):
        """Predict and score on validation set."""
        self.logger.info("\u25CF VALIDATE ON HOLDOUT")
        # use best param from cv on train (automated in GridSearch if refit=True)
        # self.via_scorers(self.scorers)
        self.via_metrics(self.p['metrics'])

    def via_scorers(self, scorers):
        # via scorers
        # upside: sklearn consistent
        # downside:
        #   requires multiple inference
        #   non-score metric not possible (confusion_matrix)

        for name, scorer in scorers.items():
            self.logger.log(25, f"{name}:")
            self.logger.log(5, f"{name}:")
            # result score on Train
            score_train = scorer(self.estimator, self.x_train, self.y_train)
            # result score on test
            score_test = scorer(self.estimator, self.x_test, self.y_test)
            self.logger.log(25, f"Train:\n    {score_train}\n"
                                f"Test:\n    {score_test}")
            self.logger.log(5, f"Train:\n    {score_train}\n"
                               f"Test:\n    {score_test}")
        # non-score metrics
        add_metrics = {name: self.p['metrics'][name] for name in self.p['metrics'] if name not in scorers}
        self.via_metrics(add_metrics)

    def via_metrics(self, metrics):
        # via metrics
        #   upside: only one inference
        #   donwside: errpr-prone
        #   strange: xgboost lib contain auto detection y_type logic.
        if hasattr(self.estimator, 'predict_proba'):
            th_ = self.estimator.get_params().get('estimate__apply_threshold__threshold', 0.5)
            y_pred_proba_train = self.estimator.predict_proba(self.x_train)
            y_pred_train = self.prob_to_pred(y_pred_proba_train, th_)
            y_pred_proba_test = self.estimator.predict_proba(self.x_test)
            y_pred_test = self.prob_to_pred(y_pred_proba_test, th_)
        else:
            y_pred_proba_train = None
            y_pred_proba_test = None
            y_pred_train = self.estimator.predict(self.x_train)
            y_pred_test = self.estimator.predict(self.x_test)

        for name, metric in metrics.items():
            if metric[1].get('needs_proba', False):
                if not hasattr(self.estimator, 'predict_proba'):
                    raise TypeError("Estimator object has no attribute 'predict_proba'")
                y_pred_train_curr = y_pred_proba_train[:, self.pos_label_ind]
                y_pred_test_curr = y_pred_proba_test[:, self.pos_label_ind]
            elif metric[1].get('needs_threshold', False):
                y_pred_train_curr = self.estimator.decision_function(self.x_train)
                y_pred_test_curr = self.estimator.decision_function(self.x_test)
            else:
                y_pred_train_curr = y_pred_train
                y_pred_test_curr = y_pred_test

            self.logger.log(25, f"{name}:")
            self.logger.log(5, f"{name}:")
            # skip make_scorer params
            kw_args = {key: metric[1][key] for key in metric[1]
                       if key not in ['greater_is_better', 'needs_proba',
                                      'needs_threshold', 'needs_custom_kw_args']}

            if metric[1].get('needs_custom_kw_args', False):
                if self.estimator.steps[0][1].kw_args:
                    kw_args.update(self.estimator.steps[0][1].kw_args)

            # result score on Train
            score_train = metric[0](self.y_train, y_pred_train_curr, **kw_args)
            # result score on test
            score_test = metric[0](self.y_test, y_pred_test_curr, **kw_args)

            # [deprecated] not so pretty
            # if metric[0].__name__ == 'confusion_matrix':
            #     labels = metric[1].get('labels', self.classes_)
            #     score_train = pretty_confusion_matrix(score_train, labels=labels, prefix='    ')
            #     score_test = pretty_confusion_matrix(score_test, labels=labels, prefix='    ')
            if metric[0].__name__ == 'confusion_matrix':
                labels = metric[1].get('labels', self.classes_)
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

    # =============================================== dump ==========================================================
    def dump(self):
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
        # dump to disk in models dir
        dirpath = '{}/results/models'.format(self.project_path)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        file = f"{dirpath}/{self.p_hash}_{self.data_hash}_dump.model"
        if not os.path.exists(file):
            # prevent double dumping
            joblib.dump(self.estimator, file)
            self.logger.log(25, 'Save fitted model to file:\n  {}'.format(file))
        else:
            self.logger.warning('Warnning: skip dump: model file already exists\n    {}\n'.format(file))

        # alternative:
        # with open(file, 'wb') as f:
        #     pickle.dump(self.estimator, f)
        return file

    # =============================================== load ==========================================================
    def load(self, file):
        """Load fitted model on disk/string.

        Note:
            Better use only the same version of sklearn.

        """
        self.logger.info("\u25CF LOAD MODEL")
        self.estimator = joblib.load(file)
        self.logger.info('Load fitted model from file:\n    {}'.format(file))

        # alternative
        # with open(f"{self.project_path}/sump.model", 'rb') as f:
        #     self.estimator = pickle.load(f)

    # =============================================== predict ==========================================================
    # @memory_profiler
    def predict(self, data, raw_names, estimator=None):
        """Predict on new data.

        Args:
            data (pd.DataFrame): data ready for workflow unification.
            raw_names (dict): {'index': 'index_names', 'targets': 'target_names', 'feature_names'}.
            estimator (sklearn-like estimator, optional (default=None)): fitted estimator,
                if None use from workflow object.

        """
        self.logger.info("\u25CF PREDICT ON TEST")

        raw_index_names = raw_names['index']
        raw_targets_names = raw_names['targets']
        if estimator is None:
            estimator = self.estimator
        data_df, _, _ = self.unify_data(data)
        x_df = data_df.drop(['targets'], axis=1)  # was used for compatibility with unifier
        y_pred = estimator.predict(x_df.values)
        y_pred_df = pd.DataFrame(index=data_df.index.values,
                                 data={raw_targets_names[0]: y_pred}).rename_axis(raw_index_names)

        # hash of data
        data_hash = pd.util.hash_pandas_object(data_df).sum()
        # dump to disk in predictions dir
        dirpath = '{}/results/models'.format(self.project_path)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        filepath = f"{dirpath}/{self.p_hash}_{data_hash}_predictions.csv"

        with open(filepath, 'w', newline='') as f:
            y_pred_df.to_csv(f, mode='w', header=True, index=True, sep=',', line_terminator='\n')  # only LF

        self.logger.log(25, "Save predictions for new data to file:\n    {}".format(filepath))

    # =============================================== gui param ========================================================
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
