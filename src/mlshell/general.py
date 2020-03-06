"""ML workflow class
TODO:
    make test run auto how to prevent?
TODO:
    encoder step in pipeline, add choose possibility
    # ('encode_features',    mlshell.custom.encoder(encoder=sklearn.ensemble.RandomTreesEmbedding(n_estimators=300, max_depth=9), skip=True)),
TODO:
    gui сырой, пока лучше не выкладывать, или с большими оговорками
TODO:
    for custon "score" we don`t know how to calculate score_vector, it is possible to specify in conf function
    currently replace with r2 for regression
TODO
    gs_flag = False gen gui params
    check gui (i remain original index, don`t remeber where i use it, maybe x)
TODO:
    model dump, add file to append description of models (will autoerase if delete models) to fast identify and sorting
    how many data was used, score, hp
TODO:
    add multitarget  support
    add multilabel   support
TODO: встроенный CV для линейных моделей и out-of-bag можно использовать для первичного выбора диапозона параметров
TODO: добавить графиков
    https://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_with_cross_validation.html#sphx-glr-auto-examples-feature-selection-plot-rfe-with-cross-validation-py
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_predict.html#sphx-glr-auto-examples-model-selection-plot-cv-predict-py
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html#sphx-glr-auto-examples-model-selection-plot-multi-metric-evaluation-py
    "Постройте два графика оценок точности +- их стандратного отклонения в зависимости от гиперпараметра,
    убедитесь что вы действительно нашли её максимум,
    обратите внимание на большую дисперсию получаемых оценок (уменьшить её можно увеличением числа фолдов cv)."
    validation_curve для проверки переобучения.
    learning_curve для проверки сложности модели/необходимости добавить данные
    https://scikit-learn.org/stable/modules/learning_curve.html
TODO: причесать EDA
TODO: каждый GS с несколькоми эпохами с разным random state использовать, не будем попадать на выбросы
TODO: добавить проверку что в каждом фолде есть примеры обоих классов
TODO: все модели сохранять в temp необученными с индексом, чтобы если что извлечь по логу

TODO: gs надо запускать воркерами, чтобы при падении промежуточные результаты не терять
    мастер один раз заготаливает данные,
    GS в мастере, раскидывает по воркерам
    воркер для каждого фита должен создавать файл, чтобы не потерять в случае чего
    который мастер мержит для анализа
TODO: добавь сообщение чтобы выводил если папка cache слишком большая

Note:
    для kaggle train не надо делить на train-test, валидация на паблике

"""

import mlshell.custom
import mlshell.default
from mlshell.libs import *


def check_hash(function_to_decorate):
    """Decorator to check alteration in hash(self.data_df) after call method"""
    def wrapper(*args, **kwargs):
        self = args[0]
        before = pd.util.hash_pandas_object(self.data_df).sum()
        function_to_decorate(*args, **kwargs)
        after = pd.util.hash_pandas_object(self.data_df).sum()
        assert(before == after)
    return wrapper


class Workflow(object):
    """Class for ml workflow."""

    def __init__(self, project_path, logger,  data, params=None):
        """Initialize workflow object

        Args:
            project_path (str): path to project dir.
            logger (logging.Logger): logger object.
            data (:py:class:`pandas.DataFrame`): object (save original row index after deletes row, need reindex).
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

            * 'feature_categor_<name>': any dtype, include binary

                enumerated, order is not important
            * 'feature_<name>': any dtype, exclude binary

                numerated, order is important
            * 'targets': any dtype

                for classification should be binary, ordinalencoded
                pos label should be > others in sorting np.unique(y)

        """
        self.project_path = project_path
        self.logger = logger
        # use default if skipped in params
        temp = mlshell.default.DEFAULT_PARAMS
        if params is not None:
            temp.update(params)
        self.p = temp
        # hash of hp_params
        self.p_hash = md5(str(self.p).encode('utf-8')).hexdigest()

        if self.p['estimator_type'] == 'classifier':
            self.classes_ = np.unique(data['targets'])
            self.n_classes = self.classes_.shape[0]
            self.neg_label = self.classes_[0]
            self.pos_label_ind = np.where(self.classes_ == params['pos_label'])[0][0]

        self.check_data_format(data, self.p)
        self.np_error_stat = {}
        np.seterrcall(self.np_error_callback)

        self.data_df, self.categoric_ind_name, self.numeric_ind_name = self.unifier(data)
        # hash of data before split
        self.data_hash = pd.util.hash_pandas_object(self.data_df).sum()
        # calc unique values (np.nan not included as value)
        self.value_counts = {column_name: self.data_df[column_name].value_counts()
                             for i, column_name in enumerate(self.data_df.columns)}
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
        self.best_params_ = None
        self.modifiers = []
        # fulfill in self.gen_gui_params()
        self.gui_params = {}

    def check_data_format(self, data, params):
        """check data format"""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("input data should be pandas.DataFrame object")
        if 'targets' not in data.columns:
            raise KeyError("input dataframe should contain 'targets' column")
        if not all(['feature_' in column for column in data.columns if 'targets' not in column]):
            raise KeyError("all name of dataframe features columns should start with 'feature_'")
        if params['estimator_type'] == 'classifier':
            if self.n_classes > 2:
                raise ValueError('only binary classification with pos_label={}'.format(params['pos_label']))
            if params['pos_label'] != self.classes_[-1]:
                raise ValueError("pos_label={} should be last in np.unique(targets), current={}"
                                 .format(params['pos_label'], self.classes_))

    def set_params(self, params):
        """[Deprecated] Set user param attributes

        Args:
            params (dic): {'attribute_name':value}.

        """
        for k, v in params.items():
            setattr(self, k, v)
        self.logger.info('User params:\n    {}\n'.format(jsbeautifier.beautify(str(params))))

    def np_error_callback(self, *args):
        """Numpy errors handler, count errors by type"""
        if args[0] in self.np_error_stat.keys():
            self.np_error_stat[args[0]] += 1
        else:
            self.np_error_stat[args[0]] = 1

    # =============================================== unify ============================================================
    def unifier(self, data):
        """ unify input dataframe

        Note:
            * delete duplicates, (not reset index, otherwise problem with base_plot)
            * log info about gaps
            * unify gaps
                * if gap in targets => raise MyException
                * if gap in categor => 'unknown'(auto change dtype) => ordinalencoder
                * if gap in non-categor => np.nan
            * transform to np.float64 (python float = np.float = np.float64 = C double = np.double(if 64 bit processor))
            * define dics for
                * self.categoric_ind_name => {1:('feat_n', ['cat1', 'cat2'])}
                * self.numeric_ind_name   => {2:('feat_n',)}
                * self.value_counts       => {'feat_n':uniq_values}

        Returns:
            data (pd.DataFrame): unified input dataframe
            categoric_ind_name (dict): {column_index: ('feature_categr__name',['B','A','C']),}
            numeric_ind_name (dict):  {column_index: ('feature__name',),}

        """
        # find duplicates rows
        mask = data.duplicated(subset=None, keep='first')  # duplicate rows index
        self.logger.warning('MyWarning: {} duplicates rows found'.format(np.sum(mask)))
        # count unique duplicated rows
        rows_count = data[mask].groupby(data.columns.tolist())\
            .size().reset_index().rename(columns={0: 'count'})
        rows_count.sort_values(by=['count'], axis=0, ascending=False, inplace=True)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            self.logger.debug('Duplicates found\n{}\n'
                              .format(tabulate.tabulate(rows_count, headers='keys', tablefmt='psql')))

        if self.p['del_duplicates']:
            # delete duplicates, (not reset index, otherwise problem with base_plot)
            size_before = data.size
            data.drop_duplicates(keep='first', inplace=True)
            # data.reset_index(drop=True, inplace=True) problem with base_plot
            size_after = data.size
            if size_before - size_after != 0:
                self.logger.warning('MyWarning: delete duplicates rows ({} values)\n'.format(size_before - size_after))

        # calculate amount of gaps
        gaps_number = data.size - data.count().sum()
        # log
        gaps_number_dic = {}
        if gaps_number > 0:
            gaps_number_dic = {column_name: data[column_name].size - data[column_name].count()
                               for column_name in data}
            self.logger.warning('MyWarning: gaps:{} {}%\n'.format(gaps_number, gaps_number / data.size))
            self.logger.warning(gaps_number_dic)

        categoric_ind_name = {}
        numeric_ind_name = {}
        for ind, column_name in enumerate(data):
            if 'targets' in column_name:
                if column_name in gaps_number_dic:
                    raise MyException("MyError: gaps in targets")
                    # delete rows with gaps in targets
                    # data.dropna(self, axis=0, how='any', thresh=None, subset=[column_name],inplace=True)
                continue
            if '_categor_' in column_name:
                # fill gaps with 'unknown'
                data[column_name].fillna(value='unknown', method=None, axis=None,
                                         inplace=True, limit=None, downcast=None)
                # encode
                encoder = sklearn.preprocessing.OrdinalEncoder(categories='auto')
                data[column_name] = encoder.fit_transform(data[column_name].
                                                          values.reshape(-1, 1))
                categoric_ind_name[ind-1] = (column_name,
                                             encoder.categories_[0])  # ('feature_categor__name',['B','A','C'])
            else:
                # fill gaps with np.nan
                data[column_name].fillna(value=np.nan, method=None, axis=None,
                                         inplace=True, limit=None, downcast=None)
                numeric_ind_name[ind-1] = (column_name,)

        # cast to np.float64 without copy
        data = data.astype(np.float64, copy=False)

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
        if self.p['isneed_cache']:
            cachedir = f"{self.project_path}/temp"
            # delete cache if nessesery
            if self.p['cache_update'] and os.path.exists(cachedir):
                shutil.rmtree(cachedir, ignore_errors=True)
            self.logger.debug('Cachedir:\n    {}\n'.format(cachedir))
            if not os.path.exists(cachedir):
                # create temp dir for cache if not exist
                os.makedirs(cachedir)
            else:
                self.logger.debug('Warning: use caching results \n')
        else:
            cachedir = None

        # assemble several steps that can be cross-validated together
        pipeline_ = self.pipeline_steps()
        last_step = self.create_last(self.p['main_estimator'], pipeline_)
        pipeline_.append(('estimate', last_step))
        self.estimator = sklearn.pipeline.Pipeline(pipeline_, memory=cachedir)
        # set zero position params from hp_grid
        for name, vals in self.p['hp_grid'].items():
            self.estimator.set_params(**{name: vals[0]})

        #     path = param.split('__')
        #     for step in steps:
        #         if step[0] == path[0]:
        #             step.__setattr__()
        #       #      default_steps[]

        # nice print of pipeline
        params = self.estimator.get_params()
        self.logger.debug('Pipeline steps:')
        for i, step in enumerate(params['steps']):
            step_name = step[0]
            step_hp = {key: params[key] for key in params.keys() if step_name + '__' in key}
            self.logger.debug('  ({})  {}\n    {}'.format(i, step[0], step[1]))
            self.logger.debug('    hp:\n   {}'.format(jsbeautifier.beautify(str(step_hp))))
        self.logger.debug('+' * 100)

        if self.p['debug_pipeline']:
            self.debug_pipeline_()

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
        if self.p['estimator_type'] == 'regressor':
            if pipeline_[-1][0] == 'estimate':
                transformer = pipeline_[-1][1].__dict__['transformer']
                del pipeline_[-1]
            else:
                transformer = None
            last_step = sklearn.compose.TransformedTargetRegressor(regressor=estimator,
                                                                   transformer=transformer, check_inverse=True)
        elif self.p['estimator_type'] == 'classifier':
            if pipeline_[-1][0] == 'estimate':
                del pipeline_[-1]
            if self.p['th_strategy'] == 0:
                last_step = sklearn.pipeline.Pipeline(steps=[('classifier', estimator)])
                _ = self.p['hp_grid'].pop('estimate__apply_threshold__threshold', None)
            else:
                last_step = sklearn.pipeline.Pipeline(steps=[
                        ('classifier',       mlshell.custom.PredictionTransformer(estimator)),
                        ('apply_threshold',  mlshell.custom.ThresholdClassifier(self.classes_,
                                                                                self.pos_label_ind,
                                                                                self.p['pos_label'],
                                                                                self.neg_label, threshold=0.5)),
                        ])
        else:
            raise MyException("MyError: unknown estimator type = {}".format(self.p['estimator_type']))

        if last_step._estimator_type != self.p['estimator_type']:
            raise MyException('MyError:{}:{}: wrong estimator type'.format(self.__class__.__name__,
                                                                           inspect.stack()[0][3]))
        return last_step

    def pipeline_steps(self):
        """Configure pipeline steps

        Returns:
            pipeline

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
        steps = mlshell.default.CreateDefaultPipeline(self.categoric_ind_name,
                                     self.numeric_ind_name, self.set_custom_param, self.p).default_steps
        return steps
        # sklearn.pipeline.Pipeline(steps)
        # # update with zero postion hp_grid:
        # for param in self.p['hp_grid']:
        #     path = param.split('__')
        #     for step in steps:
        #         if step[0] == path[0]:
        #             step.__setattr__()
#       #      default_steps[]

    def set_custom_param(self, *arg, **kwarg):
        """Use with FunctionTransformer to set self-attributes in GS"""
        # self.logger.debug(kwarg)
        for k, v in kwarg.items():
            setattr(self, k, v)
        return arg[0]  # pass x further

    def debug_pipeline_(self):
        """Fit estimator on whole data for debug"""
        x, y = self.tonumpy(self.data_df)
        fitted = self.estimator.fit(x, y)
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
            self.logger.warning('{0}{1}\n{0}{2}'.format('   ' * level, ob_name, ob))
            if hasattr(ob, '__dict__'):
                for attr_name in ob.__dict__:
                    attr = getattr(ob, attr_name)
                    self.logger.warning('{0}{1}\n{0}   {2}'.format('   ' * (level + indent), attr_name, attr))
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

    # =============================================== before split analyze =============================================
    @check_hash
    # @memory_profiler
    def before_split_analyze(self):
        """EDA on full dataframe

        * check and log imbalance
        * check and log variance
        * if debug: fit estimator and log detail information

        """
        x, y = self.tonumpy(self.data_df)

        self.df_info()
        self.check_imbalance()
        self.check_collinearity(x, y)
        self.check_variance(x, y)
        self.statmodel_check(x, y)
        if self.p['plot_analysis']:
            self.plottings()

    def tonumpy(self, data_df):
        """Convert dataframe to features and target numpy arrays"""
        # almost always copy
        x = data_df[data_df.columns[1:]]  # .values
        y = data_df['targets']  # .values
        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)
        return x, y

    def statmodel_check(self, x, y):
        """Trasform with pipeline and fit on ols/logit

        Args:
            x (np.array): fetures array
            y (np.array): target array

        Note:
            it possible to make custom sklearn estimator through statmodels API for CV
        """

        # add intercept (statmodels don`t auto add)
        x = np.c_[x, np.ones(x.shape[0])]

        pipeline_ = self.pipeline_steps()
        if pipeline_[-1][0] == 'estimate':
            del pipeline_[-1]

        # will use default pipline (without `estimate`  step)
        transformer = sklearn.pipeline.Pipeline(pipeline_, memory=None)
        # alternative: set zero position params from hp_grid
        # for name, vals in self.p['hp_grid'].items():
        #     if name.startswith('estimate'):
        #         continue
        #     transformer.set_params(**{name: vals[0]})
        transformer.fit_transform(x, y)

        fitted_estimator = self.fit_sm(x, y)
        if self.p['estimator_type'] == 'regressor':
            fitted_estimator, y = self.target_normalization(fitted_estimator, x, y)

        fitted_estimator, cov_type = self.handle_homo(fitted_estimator, x, y)
        self.check_leverage(fitted_estimator)

        # [deprecated] add/drop intercept
        # self.data_df=self.data_df.assign(**{'feature_intercept':np.full(self.data.shape[0],
        #     fill_value=1, dtype=np.float64, order='C')})
        # self.data['feature_intercept'] = 1
        # delete intercept from data not to influence forward analyse
        # self.data_df.drop(['feature_intercept'], axis=1, inplace=True)

    def fit_sm(self, x, y, cov_type='nonrobust'):
        """Fit statmodel"""
        if self.p['estimator_type'] == 'regressor':
            estimator = sm.OLS(y, x)
        else:
            estimator = sm.Logit(y, x)
        estimator = estimator.fit(cov_type=cov_type)
        self.logger.info('{}'.format(estimator.summary()))
        return estimator

    def target_normalization(self, estimator, x, y, cov_type='nonrobust'):
        """ Check normality for residuals, y yeojohnson transform if necessary"""
        _, p = scipy.stats.shapiro(estimator.resid)
        self.logger.info('Shapiro criteria p-value={}'.format(p))
        if p < 0.05:
            self.logger.warning('MyWarning: use Yeo-Johnson normalization for target')
            # skew = self.estimator.diagn['skew']
            # kurtosis = self.estimator.diagn['kurtosis']

            # yeojohnson transform,box-cox only for positive
            y, lmbda = scipy.stats.yeojohnson(y)  # with copy
            # refit model
            estimator = self.fit_sm(x, y)
            _, p_new = scipy.stats.shapiro(estimator.resid)
            self.logger.info('Shapiro criteria after target normalization p_value = {}'.format(p_new))
        if self.p['plot_analysis']:
            # visual check Q-Q for residuals
            plt.figure(figsize=(16, 7))
            plt.subplot(121)
            scipy.stats.probplot(estimator.resid, dist="norm", plot=plt)
            plt.subplot(122)
            plt.hist(np.log(estimator.resid))
            plt.xlabel('Residuals', fontsize=14)
            plt.show()
        return estimator, y

    def handle_homo(self, estimator, x, y, cov_type='nonrobust'):
        """Check homo"""
        p = sms.het_breuschpagan(estimator.resid, estimator.model.exog)[1]
        self.logger.info('Breusch-Pagan criteria p-value={}'.format(p))
        if p < 0.05:
            self.logger.warning('MyWarning:  apply HC1'.format(p))
            cov_type = 'HC1'
            estimator = self.fit_sm(x, y, cov_type=cov_type)
        return estimator, cov_type

    def check_leverage(self, estimator):
        """Check leverage"""
        if self.p['plot_analysis']:
            plt.figure(figsize=(8, 7))
            smg.plot_leverage_resid2(estimator)
        self.logger.debug('UNRECONSTRUCTED leverage')

    def df_info(self):
        """Log info() and describe() for data_df"""
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            self.logger.info('{}\n'.format(tabulate.tabulate(self.data_df.describe(), headers='keys', tablefmt='psql')))
            self.logger.info('{}\n'.format(self.data_df.info()))
            # self.logger.info('Memory usage{}\n'.format(self.data_df.memory_usage())) by column

    def check_imbalance(self):
        """Check and log high imbalanced features, targets"""
        # if categorical or count_values < 10
        categoric_names = [i[0] for i in self.categoric_ind_name.values()]
        numeric_names = [i[0] for i in self.numeric_ind_name.values()]
        if self.p['estimator_type'] == 'classifier':
            categoric_names += ['targets']
        else:
            numeric_names += ['targets']
        for column_name in categoric_names + numeric_names:
            column = self.value_counts[column_name]
            if column_name in numeric_names and column.size > 10:
                continue
            amounts = column.values  # amount
            uniq_values = column.index   # value
            for i, val in enumerate(amounts):
                if float(val) / column.size < 0.05:
                    self.logger.warning('MyWarning: class imbalanced: {} {:.2f}<5% {}'
                                        .format(int(uniq_values[i]), 100 * float(val) / column.sum(), column_name))
                    # [deprecated] class balance for all data
                    # self.logger.debug('{}\n'.join(map(lambda x:'{}\n{}'
                    # .format(x[0],x[1]),[(column_name, self.value_counts[column_name])
                    # for column_name in self.binary_names])))

    def check_variance(self, x, y):
        """Check and log variance for features, targets

        variance=squared deviation=np.std**2

        Args:
            x (2d numpy.array, dataframe): features
            y (numpy.array, dataframe): targets

        """
        # TODO: check differ from dataframe describe()
        if isinstance(x, pd.DataFrame):
            x = x.values
            y = y.values
        var_feat = sklearn.feature_selection.VarianceThreshold(threshold=0).fit(x)
        var_targ = sklearn.feature_selection.VarianceThreshold(threshold=0).fit(y.reshape(-1, 1))
        self.logger.info('target|features variance:\n{}|{}\n'.format(var_targ.variances_[0], var_feat.variances_))

    def check_collinearity(self, x, y):
        """"check and log collinearity analyze  for features matrix

         Args:
            x (2d numpy.array, dataframe): features
            y (numpy.array, dataframe): targets
        """
        # TODO: underconstucted
        if isinstance(x, pd.DataFrame):
            x = x.values
            y = y.values

        if True:
            self.logger.warning('UNRECONSTRUCTED collinearity')
            return
        # # VIF method
        # self.calculate_vif(self.data[self.features_names])
        # # Condition number method
        # self.calculate_condnumber(self.data[self.features_names])
        # # вроде бы само содержатся в параметре exog

    def calculate_vif(self, x, thresh=5.0):
        """Checks VIF values and then drops variables whose VIF is more than threshold.

        Args
            thres (float): threshold

        Note:
            runtime could be big

        """
        # TODO: underconstucted
        variables = list(range(x.shape[1]))
        dropped = True
        while dropped:
            dropped = False
            vif = [statsmodels.stats.outliers_influence.variance_inflation_factor(x.loc[:, variables].values, ix)
                   for ix in range(x.loc[:, variables].shape[1])]
            maxloc = vif.index(max(vif))
            if max(vif) > thresh:
                self.logger.debug('Dropping {} at index: {}'.format(x.loc[:, variables].columns[maxloc], str(maxloc)))
                del variables[maxloc]
                dropped = True
        self.logger.debug('Remaining variables:\n{}'.format(x.columns[variables]))
        return x.loc[:, variables]

    def calculate_condnumber(self, x):
        """Calculate condnumber"""
        # TODO: underconstuct
        corr = np.corrcoef(x, rowvar=False)       # correlation matrix
        eig_numbers, eig_vectors = np.linalg.eig(corr)  # eigen values & eigen vectors

    def plottings(self, indices=None):
        """Plot scatter-matrix for numeric columns in dataframe

        Args:
            indices (list): column names

        """
        # scatter-plot for continuous data
        if indices is None:
            indices = self.numeric_ind_name

        if len(indices) < 15:
            pd.plotting.scatter_matrix(self.subcolumns(self.data_df, indices=indices),
                                       alpha=0.2, figsize=(15, 15), diagonal='hist')
            plt.show()
        else:
            self.logger.warning('Too much columns for plot scatter matrix:{}'.format(len(indices)))

    # =============================================== split ==========================================================
    # @memory_profiler
    def split(self):
        """Split data on train, test

        Note:
            if `split_train_size` set to 1.0, use full dataset to CV (test=train)

        """
        if self.p['split_train_size'] == 1.0:
            train = test = self.data_df
            self.train_index = self.test_index = self.data_df.index
        else:
            train, test, self.train_index, self.test_index = sklearn.model_selection.train_test_split(
                self.data_df, self.data_df.index.values,
                train_size=self.p['split_train_size'], test_size=None,
                random_state=42, shuffle=False, stratify=None)
        columns = self.data_df.columns
        # deconcatenate without copy, better use dataframe (provide index)
        self.x_train = train[[name for name in columns if 'feature' in name]]  # .values
        self.y_train = train['targets']  # .values
        self.x_test = test[[name for name in columns if 'feature' in name]]  # .values
        self.y_test = test['targets']  # .values
        # TODO: добавить проверку что в трейне каждого фолда будут объекты обоих классов, иначе проблема c predict_proba

    # =============================================== cv ===============================================================
    def cv(self, n_splits=None):
        """Method to generate samples for cv

        Args:
            n_splits (int): number of splits

        Returns:
            object which have split method yielding train/test splits indices
        """
        if n_splits is not None:
            self.p['cv_splitter'].n_splits = n_splits

        return self.p['cv_splitter']

    # =============================================== score ============================================================
    def custom_scorer(self, estimator, x, y_true):
        """Custom scorer.

        Args:
            estimator: fitted estimator.
            x (dataframe, np.ndarray): features test.
            y_true (dataframe, np.ndarray): true targets test.

        Returns:
            score value

        Note:
            Score will be maximize.
            Have to set negative if _loss or _error metric.
            In case of built-in metric:

                scorer = metrics.make_scorer(metrics.accuracy_score, greater_is_better=True)
                score = scorer(estimator, x, y)

        """
        if isinstance(x, pd.DataFrame):
            index = y_true.index
            x = x.values
            y_true = y_true.values
        # [deprecated] need dynamic change of self.th_strategy
        # if self.estimator_type == 'classifier':
        #     if self.th_strategy == 1 and hasattr(estimator, 'predict_proba'):
        #         y_pred_proba = estimator.predict_proba(x)
        #         score = metrics.roc_auc_score(y_true, y_pred_proba[:, self.pos_label_ind])
        #     elif self.th_strategy == 2 and hasattr(estimator, 'predict_proba'):
        #         y_pred_proba = estimator.predict_proba(x)
        #         y_pred = self.brut_th_(y_true, y_pred_proba)
        #         score = self.metric(y_true, y_pred)
        #     else:
        #         y_pred = estimator.predict(x)
        #         score = self.metric(y_true, y_pred)
        # else:
        #     y_pred = estimator.predict(x)
        #     score = self.metric(y_true, y_pred)
        y_pred = estimator.predict(x)
        score = self.metric(y_true, y_pred)
        return score

    def score_strategy_1(self, estimator, x, y_true):
        """Calculate score strategy (1) for classification

        Note:
            Predict probabilities.
            Calculate roc_auc.

        """
        y_pred_proba = estimator.predict_proba(x)
        score = sklearn.metrics.roc_auc_score(y_true, y_pred_proba[:, self.pos_label_ind])
        return score

    def score_strategy_2(self, estimator, x, y_true):
        """Calculate score strategy (2) for classification

        Note:
            Predict probabilities.
            Brutforce th from roc_curve range.
            Score main metric after fix best th.

        """
        y_pred_proba = estimator.predict_proba(x)
        best_th_, _ = self.brut_th_(y_true, y_pred_proba)
        y_pred = self.prob_to_pred(y_pred_proba, best_th_)
        score = self.metric(y_true, y_pred)
        return score

    def metric(self, y_true, y_pred, meta=False):
        """Calculate metric.

        Args:
            y_true (np.ndarray): true targets.
            y_pred (np.ndarray): predicted targets.
            meta (bool): if True calculate metadata for visualization.

        Returns:
            score (float): metric score
            meta (dict): cumulative score in dynamic; TP,FP,FN in points for classification.

        """
        if self.p['estimator_type'] == 'classifier':
            return self.metric_classifier(y_true, y_pred, meta)
        else:
            return self.metric_regressor(y_true, y_pred, meta)

    def metric_classifier(self, y_true, y_pred, meta=False):
        """Сalculate classification metric for maximization.

        Detailed meta for external use cases.

        Args:
            y_true (np.ndarray): true targets.
            y_pred (np.ndarray): predicted targets.
            meta (bool): if True calculate metadata for visualization

        Returns:
            score (float): metric score
            meta (dict): cumulative score in dynamic; TP,FP,FN in points for classification.

        """
        # score
        score = self.p['metrics']['score'][0](y_true, y_pred)
        if not meta:
            return score

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
        """Сalculate regression metric for minimization.

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
        score = self.p['metrics']['score'][0](y_true, y_pred)
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

    # =============================================== gridsearch =======================================================
    @check_hash
    # @memory_profiler
    def fit(self, gs_flag=False):
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

            Alternative:
                * random sampling from param space: model_selection.RandomizedSearchCV(), GaussProcess(), Tpot()
                * estimators with regularization have self, efficient gridsearch
                * estimators with bagging have self, efficient gridsearch
                * estimators with AIC, BIC

        If gs_flag is True and hp_grid is not {}: run grid search else just fit estimator
        For regression:
            use 'score' from score_metrics as main
        For classification:
            th1: use roc_auc score on predict_proba
            th2:

        """
        if gs_flag and self.p['hp_grid']:
            self.optimize()
        else:
            self.estimator.fit(self.x_train, self.y_train)

        # dump self.estimator on disk
        if self.p['isneeddump']:
            file = self.dump()

    def optimize(self):
        # tune hp on train by cv
        # param, fold -> fit(fold_train) -> predict(fold_test) -> score for params
        scoring = self.get_scoring()
        n_iter = self.get_n_iter()
        pre_dispatch = self.get_pre_dispatch()

        # optimize score
        optimizer = sklearn.model_selection.RandomizedSearchCV(
            self.estimator, self.p['hp_grid'], scoring=scoring, n_iter=n_iter,
            n_jobs=self.p['n_jobs'], pre_dispatch=pre_dispatch, iid=False,
            refit='score', cv=self.cv(), verbose=self.p['gs_verbose'], error_score=np.nan,
            return_train_score=True).fit(self.x_train, self.y_train)
        self.estimator = optimizer.best_estimator_
        self.best_params_ = optimizer.best_params_
        # nice print
        self.gs_print(optimizer)
        # dump CV results on disk
        self.dump_runs(optimizer)

        # [experimentary] optimize threshold if necessary
        if self.p['estimator_type'] == 'classifier' and (self.p['th_strategy'] == 1 or self.p['th_strategy'] == 2):
            scoring = self.metrics_to_scorers(self.p['metrics'])
            th_range, predict_proba, y_true = self.calc_th_range()
            optimizer_th_ = sklearn.model_selection.RandomizedSearchCV(
                mlshell.custom.ThresholdClassifier(self.classes_, self.pos_label_ind,
                                                   self.p['pos_label'], self.neg_label),
                {'threshold': th_range}, n_iter=th_range.shape[0],
                scoring=scoring,
                n_jobs=1, pre_dispatch=2, iid=False, refit='score', cv=self.cv(),
                verbose=1, error_score=np.nan, return_train_score=True).fit(predict_proba, y_true)
            best_th_ = optimizer_th_.best_params_['threshold']
            self.best_params_['estimate__apply_threshold__threshold'] = best_th_
            self.modifiers.append('estimate__apply_threshold__threshold')
            self.p['hp_grid']['estimate__apply_threshold__threshold'] = th_range
            # refit with treshold
            self.estimator.set_params(**self.best_params_)
            self.estimator.fit(self.x_train, self.y_train)
            self.logger.info('CV best threshold:\n    {}'.format(best_th_))
            # TODO: combine with gs_print
            # TODO: add dump result

    def metrics_to_scorers(self, metrics):
        """Make from scorers from metrics

        Args:
            metrics (dict): {'name': (sklearn metric object, bool greater_is_better), }

        Returns
            scorers (dict): {'name': sklearn scorer object, }

        """
        scorers = {}
        for name, metric in metrics.items():
            scorers[name] = sklearn.metrics.make_scorer(metric[0], greater_is_better=metric[1])
        return scorers

    def get_scoring(self):
        """Set gs target score for different approaches"""
        scoring = self.metrics_to_scorers(self.p['metrics'])
        if self.p['estimator_type'] == 'classifier':
            if self.p['th_strategy'] == 1:
                scoring = {'score': self.score_strategy_1}
            elif self.p['th_strategy'] == 2:
                scoring = {'score': self.score_strategy_2}
            elif self.p['th_strategy'] == 3:
                scoring = self.metrics_to_scorers(self.p['metrics'])
                if 'estimate__apply_threshold__threshold' in self.p['hp_grid']:
                    self.logger.warning('MyWarning: brutforce th_ 3.1')
                else:
                    th_range, _, _ = self.calc_th_range()
                    self.p['hp_grid'].update({'estimate__apply_threshold__threshold': th_range})
                    self.logger.warning('MyWarning: brutforce threshold strategy 3.2')
        return scoring

    def get_n_iter(self):
        """Set gs number of runs"""
        # calculate from hps ranges if user 'runs' is not given
        if self.p['runs'] is None:
            n_iter = np.prod([len(i) if isinstance(i, list) else i.shape[0] for i in self.p['hp_grid'].values()])
        else:
            n_iter = self.p['runs']
        return n_iter

    def get_pre_dispatch(self):
        """Set gs parallel jobs

        If n_jobs was set to a value higher than one, the data is copied for each parameter setting.
        Using pre_dispatch you can set how many pre-dispatched jobs you want to spawn.
        The memory is copied only pre_dispatch many times. A reasonable value for pre_dispatch is 2 * n_jobs.

        """
        # n_jobs can be -1, pre_dispatch=1 mean  spawn all
        pre_dispatch = max(2, self.p['n_jobs']) if self.p['n_jobs'] else 1
        return pre_dispatch

    # @time_profiler
    def dump_runs(self, res):
        """Dumps grid search results in <timestamp>_runs.csv

        Args:
            res (sklearn optimizer): contain GS reults.

        Note:
            _runs.csv contain columns:

                * all estimator parameters.
                * 'id' random UUID for one run (hp combination).
                * 'data_hash' pd.util.hash_pandas_object hash of data before split.
                * 'params_hash' user params md5 hash (cause of function memory address will change at each workflow).
                * 'estimator_type' regressor or classifier.
                * 'cv_splitter'.
                * 'split_train_size'.
                * 'rows_limit'.
        """
        # get full params for each run
        runs = len(res.cv_results_['params'])
        lis = list(range(runs))
        est_clone = sklearn.clone(self.estimator)  # not clone attached data, only params
        for i, param in enumerate(res.cv_results_['params']):
            est_clone.set_params(**param)
            lis[i] = est_clone.get_params()
        df = pd.DataFrame(lis)  # too big ro print
        # merge df with res.cv_results_ with replace (exchange args if don`t need replace)
        # cv_results consist suffix param_
        param_labels = set(i for i in res.cv_results_.keys() if 'param_' in i)
        other_labels = set(res.cv_results_.keys())-param_labels
        update_labels = set(df.columns).intersection(other_labels)
        df = pd.merge(df, pd.DataFrame(res.cv_results_).drop(list(param_labels), axis=1),
                      how='outer', on=list(update_labels), left_index=True, right_index=True,
                      suffixes=('_left', '_right'))
        # pipeline
        # df = pd.DataFrame(res.cv_results_)
        # rows = df.shape[0]
        # df2 = pd.DataFrame([self.estimator.get_params()])

        # unique id for param combination
        run_id_list = [str(uuid.uuid4()) for _ in range(runs)]

        df['id'] = run_id_list
        df['data_hash'] = self.data_hash
        df['estimator_type'] = self.p['estimator_type']
        df['estimator_name'] = self.p['main_estimator'].__class__.__name__
        df['cv_splitter'] = self.p['cv_splitter']
        df['split_train_size'] = self.p['split_train_size']
        df['rows_limit'] = str((self.p['rows_limit'], self.p['random_skip']))
        df['params_hash'] = self.p_hash
        # df=df.assign(**{'id':run_id_list, 'data_hash':data_hash_list, 'estimator_type':es })

        # cast to string before dump and print, otherwise it is too long
        # alternative: json.loads(json.dumps(data)) before create df
        object_labels = list(df.select_dtypes(include=['object']).columns)
        df[object_labels] = df[object_labels].astype(str)

        # dump to disk in run dir
        dirpath = '{}/runs'.format(self.project_path)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        filepath = '{}/runs/{}_runs.csv'.format(self.project_path, int(time.time()))
        with open(filepath, 'a', newline='') as f:
            df.to_csv(f, mode='a', header=f.tell() == 0, index=False, line_terminator='\n')
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

    def gs_print(self, res):
        """nice print"""

        # inputs
        self.logger.info('hp grid:\n    {}'.format(jsbeautifier.beautify(str(self.p['hp_grid']))))
        # find varied hp
        self.modifiers = []
        for key, val in self.p['hp_grid'].items():
            if isinstance(val, list):
                size = len(val)
            else:
                size = val.shape[0]
            if size > 1:
                self.modifiers.append(key)
        param_modifiers = set('param_'+i for i in self.modifiers)

        # outputs
        runs_avg = {'mean_fit_time': res.cv_results_['mean_fit_time'].mean(),
                    'mean_score_time': res.cv_results_['mean_score_time'].mean()}
        df = pd.DataFrame(res.cv_results_)[[key for key in res.cv_results_ if key in param_modifiers
                                            or 'mean_train' in key or 'mean_test' in key]]
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            # self.logger.debug('{}'.format(df.head()))
            self.logger.info('{}'.format(tabulate.tabulate(df, headers='keys', tablefmt='psql')))
        # Alternative: df.to_string()

        self.logger.info('GridSearch result:\n    {}'.format(runs_avg))
        self.logger.info('CV best modifiers:\n    {}'.format({key: res.best_params_[key] for key in self.modifiers}))
        self.logger.info('CV best hp:\n    {}'.format(res.best_params_))
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
        # self.dic_flatter(dic, dic_flat)
        # pd.DataFrame(dic_flat)

    # =============================================== threshold calcs ==================================================
    def calc_th_range(self, plot_flag=True):
        """ Сalculate th range for GS strategy (1), (2), (3)

        [current implementation (1)]
            * binary target
            * fit estimator with default hp => find best_th
            * in GS th range: np.linspace(best_th/100, best_th, 10)

        Args:
            plot_flag (bool): True if need plot

        Returns:
            object which have split method yielding train/test splits indices

        Note:
            th зависит от hp
            th на каждом фолде свой
            th имеет неравномерные границы изменения, которые можно оценить из ROC curve по predict_proba

            (0) don`t use th (большинство похоже делают так)

                * не все классификаторы выдают веростности (SVM)
                * f1, logloss оптимизируют порог

            (1) (не замораиваясь обычно делают так)
                auc-roc в качестве score
                не кросс-валидировать пороги одновременно с gp
                если выбрать score например auc-roc генерализирующая способность hp будет учитываться для всех порогов
                тоже справедливо для других метрик.
                потом уже порог подбирать для лучших относительно конечного score
            (2) на каждом фолде для данной комбинации hp максимизировать score по порогу,
                при этом на фолдах будут разные порог, но генерализирующую способность эстиматора данных hp мы оценим.
                потом для лучших hp после GS, отдельным циклом выбрать на CV единый порог
            (3) с помощью метажэстиматора перебирать в заранее фиксированном диапазоне

                * заранее неизвестен дипазон th.

                    (3.1) можно взять проивольные

                        точный подбор th сильно сэкономит время только если много данных,
                        а так хоть примерный диапазон значений

                    (3.2) можно взять характерные значения th из сцепки по фолдам, что не учтет влияние гиперпараметров

                * проводится переобучение вместо того чтобы рабоать с конечным результатом.

                    кэш отчасти поможет
                    лучше бы внутренний цикл отдельный при фиксированных гиперпараметрах и трансформированных данных
                * фактически когда пойдет много FP можно останавливаться, что нельзя сделать для в брутфорсе
                * можно определить лучшие hp не заморачиваясь о пороге, для лучших hp, более детально оценить порог

            (4) [нарушает workflow] full наиболее логичный, сливать вместе roc crve диапазоны th разных фолдов

                *  GS запускать на CV=1, внутри функции score поместить CV
                * выбор оптимальных для всех фолдов порогов с учетом ROC curve на предсказании

        TODO:
            проверить кагглом

        """
        predict_proba, indices, y_true = self.cross_val_predict(
            self.estimator, self.x_train, y=self.y_train, groups=None,
            cv=self.cv(), fit_params=None, method='predict_proba')
        fpr, tpr, th_ = sklearn.metrics.roc_curve(
            y_true, predict_proba[:, self.pos_label_ind],
            pos_label=self.p['pos_label'])
        best_th_, q = self.brut_th_(y_true, predict_proba)
        th_range = self.coarse_th_range(best_th_, th_)
        if plot_flag:
            plt.plot(th_, q, 'green')
            plt.vlines(best_th_, np.min(q), np.max(q))
            plt.vlines(th_range, np.min(q), np.max(q), colors='b', linestyles=':')
            # plt.plot(th_, fpr, 'red')
            plt.show()
        # TODO: добавить график
        #     https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py
        # TODO: GS тоже можно переместить сюда
        return th_range, predict_proba, y_true

    def cross_val_predict(self, *args, **kwargs):
        """Function to make bind predict/predict_proba from folds.

        Args

        Returns:
            folds_predict_proba (2d array): probability predictions [n_test_samples x n_classes]
            folds_test_index: test samples indices

        """
        debug = False
        estimator = args[0]
        x = args[1]
        y = kwargs['y']
        cv = kwargs['cv']
        temp_pp = None
        temp_ind = None
        # TODO: в каком-то фолде может не хватать классов.
        try:
            folds_predict_proba = sklearn.model_selection.cross_val_predict(*args, **kwargs)
            folds_test_index = np.arange(0, folds_predict_proba.shape[0])
            if debug:
                temp_pp = folds_predict_proba
                temp_ind = folds_test_index
                raise ValueError('debug')
        except ValueError as e:
            print('MyWarning: {}'.format(e))
            # for TimeSplitter no prediction at first fold
            folds_predict_proba = []  # list(range(self.cv_n_splits))
            folds_test_index = []  # list(range(self.cv_n_splits))
            # th_ = [[2, 1. / self.n_classes] for i in self.classes_]  # init list for th_ for every class
            ind = 0
            for fold_train_index, fold_test_index in cv.split(x):
                if hasattr(x, 'loc'):
                    # stackingestimator__sample_weight=train_weights[fold_train_subindex]
                    estimator.fit(x.loc[fold_train_index], y.loc[fold_train_index])
                    fold_predict_proba = estimator.predict_proba(
                        x.loc[fold_test_index])  # in order of self.estimator.classes_
                else:
                    estimator.fit(x[fold_train_index], y[
                        fold_train_index])  # stackingestimator__sample_weight=train_weights[fold_train_subindex]
                    fold_predict_proba = estimator.predict_proba(
                        x[fold_test_index])  # in order of self.estimator.classes_
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
        """Get most possible th range"""
        th_range = np.linspace(max(best_th_ / 100, np.min(th_)), min(best_th_ * 2, np.max(th_)), 100)
        return th_range

    def brut_th_(self, y_true, y_pred_proba):
        # TODO: можно на основе self.metric перебирать, стопорить если пошло падение, для ускорения
        fpr, tpr, th_ = sklearn.metrics.roc_curve(
            y_true, y_pred_proba[:, self.pos_label_ind],
            pos_label=self.p['pos_label'], drop_intermediate=True)
        q = np_divide(tpr, fpr+tpr)  # tpr/(fpr+tpr)
        best_th_ = th_[np.argmax(q)]
        return best_th_, q

    def prob_to_pred(self, y_pred_proba, th_):
        """Fix threshold on predict_proba"""
        y_pred = np.where(y_pred_proba[:, self.pos_label_ind] > th_, [self.p['pos_label']], [self.neg_label])
        return y_pred

    # =============================================== validate =========================================================
    # @memory_profiler
    def validate(self):
        """Predict and score on validation set."""
        # fix best param from cv on train (automated in GridSearch if refit=True)
        y_pred_train = self.estimator.predict(self.x_train)
        y_pred_test = self.estimator.predict(self.x_test)
        for name, metric in self.p['metrics'].items():
            # if name is 'score': continue
            # result score on Train
            score = metric[0](self.y_train, y_pred_train)
            self.logger.critical('Train {}:\n    {}'.format(name, score))

            # result score on test
            score = metric[0](self.y_test, y_pred_test)
            self.logger.critical('Test {}:\n    {}'.format(name, score))
        self.logger.critical('+'*100)

        # [deprecated] via scorer
        # # result score on Train
        # score = self.score(self.estimator, self.x_train, self.y_train)
        # self.logger.info('Train {}:\n    {}'.format('score', score))

        # # result score on test
        # score = self.score(self.estimator, self.x_test, self.y_test)
        # self.logger.info('Test {}:\n    {}'.format('score', score))

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
        # dump to disk in models dir
        dirpath = '{}/models'.format(self.project_path)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        file = f"{dirpath}/{self.p_hash}_{self.data_hash}_dump.model"
        if not os.path.exists(file):
            # prevent double dumping
            joblib.dump(self.estimator, file)
            self.logger.info('Save fitted model to file\n  {}\n'.format(file))
        else:
            self.logger.warning('Model file already exists\n  {}\n'.format(file))

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
        self.estimator = joblib.load(file)
        self.logger.info('Load fitted model from file\n  {}\n'.format(file))

        # alternative
        # with open(f"{self.project_path}/sump.model", 'rb') as f:
        #     self.estimator = pickle.load(f)

    # =============================================== predict ==========================================================
    # @memory_profiler
    def predict(self, data, raw_targets_names, raw_index_names, estimator=None):
        """Predict on new data."""
        if estimator is None:
            estimator = self.estimator
        data_df, _, _ = self.unifier(data)
        x_df = data_df.drop(['targets'], axis=1)  # was used for compatibility with unifier
        y_pred = estimator.predict(x_df.values)
        y_pred_df = pd.DataFrame(index=data_df.index.values,
                                 data={raw_targets_names[0]: y_pred}).rename_axis(raw_index_names)

        # hash of data
        data_hash = pd.util.hash_pandas_object(data_df).sum()
        # dump to disk in predictions dir
        dirpath = '{}/models'.format(self.project_path)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        filepath = f"{dirpath}/{self.p_hash}_{data_hash}_predictions.csv"

        with open(filepath, 'w', newline='') as f:
            y_pred_df.to_csv(f, mode='w', header=True, index=True, sep=',', line_terminator='\n')  # only LF

        self.logger.info("Made prediction for new data:    \n{}".format(filepath))

    # =============================================== GUI param ========================================================
    def gen_gui_params(self):
        """Prepare params for visualization."""
        # rearange nested hp params
        hp_grid_flat = {}
        for key, val in self.p['hp_grid'].items():
            if key not in self.modifiers:  # only if multiple values
                continue
            if isinstance(val[0], dict):  # functiontransformer
                dic = {tuple([key, key_]): np.zeros(len(val), dtype=np.float64, order='C') for key_ in val[0].keys()}
                for i, item in enumerate(val):
                    for key_ in dic:
                        dic[key_][i] = (item[key_[1]])
                hp_grid_flat.update(dic)
            else:
                hp_grid_flat[key] = self.to_numpy(val)

        # not necessary
        best_params_flat = {}
        self.dic_flatter(self.best_params_, best_params_flat)

        self.gui_params = {
            'estimator_type': self.p['estimator_type'],
            'data': self.data_df,
            'train_index': self.train_index,
            'test_index': self.test_index,
            'estimator': self.estimator,
            'hp_grid': self.p['hp_grid'],               # {'param':range,}
            'best_params_': self.best_params_,          # {'param':value,}
            'hp_grid_flat': hp_grid_flat,               # {'param':range,}
            'best_params_flat': best_params_flat,       # {'param':value,}
            'metric': self.metric,
        }

    def dic_flatter(self, dic, dic_flat, keys_lis_prev=None):
        """Flatten the dict.

        {'a':{'b':[], 'c':[]},} =>  {('a','b'):[], ('a','c'):[]}

        Args:
            dic (dict): input dictionary
            dic_flat (dict): result dictionary
            keys_lis_prev: need for recursion

        """
        if keys_lis_prev is None:
            keys_lis_prev = []
        for key, val in dic.items():
            keys_lis = keys_lis_prev[:]
            keys_lis.append(key)
            if isinstance(val, dict):
                self.dic_flatter(val, dic_flat, keys_lis)
            else:
                if len(keys_lis) == 1:
                    dic_flat[keys_lis[0]] = self.to_numpy(val)
                else:
                    dic_flat[tuple(keys_lis)] = self.to_numpy(val)

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


if __name__ == 'main':
    pass
