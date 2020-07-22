"""Common EDA techniques."""

import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
import statsmodels.graphics.regressionplots as smg
import statsmodels.stats.outliers_influence
import numpy as np
import scipy.stats


class EDA(object):
    """Common EDA techniques."""
    def __init__(self, project_path, logger=None, params=None):
        self.project_path = project_path
        if logger is None:
            self.logger = logging.Logger('Workflow')
        else:
            self.logger = logger
        self.logger.info("\u25CF EDA")
        self.p = params
        # filled in .run_analyze()
        self.categoric_ind_name = {}
        self.numeric_ind_name = {}
        self.plot_flag = False

    def analyze_data(self, data_df, pipeline=None, categoric_ind_name=None,
                     numeric_ind_name=None, plot_flag=False):
        """EDA on dataframe.

        Note:

            * dataframe info.
            * check and log imbalance.
            * check and log variance.
            * check collinearity.
            * check variance.
            * run statmodel on whole data (after transformation if pipeline provided).
            * plot scatter matrix.

        """
        before = pd.util.hash_pandas_object(data_df).sum()
        self.plot_flag = plot_flag
        if not categoric_ind_name or not numeric_ind_name:
            self.get_ind_name(data_df)
        else:
            self.categoric_ind_name = categoric_ind_name
            self.numeric_ind_name = numeric_ind_name

        x, y = self.tonumpy(data_df)

        self.df_info(data_df)
        self.check_imbalance(data_df)
        self.check_collinearity(x, y)
        self.check_variance(x, y)
        self.statmodel_check(x, y, pipeline)
        if self.plot_flag:
            self.plottings(data_df)

        after = pd.util.hash_pandas_object(data_df).sum()
        assert(before == after)

    def tonumpy(self, data_df):
        """Convert dataframe to features and target numpy arrays"""
        # almost always copy
        x = data_df[data_df.columns[1:]]  # .values
        y = data_df['targets']  # .values
        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)
        return x, y

    def get_ind_name(self, data_df):
        for ind, column_name in enumerate(data_df):
            if 'targets' in column_name:
                continue
            if '_categor_' in column_name:
                # loose categories true names
                self.categoric_ind_name[ind - 1] = (column_name, np.unique(data_df[column_name]))
            else:
                self.numeric_ind_name[ind - 1] = (column_name,)

    def statmodel_check(self, x, y, pipeline):
        """Trasform with pipeline and fit on ols/logit

        Args:
            x (np.array): fetures array
            y (np.array): target array
            pipeline (sklearn.pipeline.Pipeline, optional):
                if provided fit_transform data, then statmodel at last step (ignore origina last step).

        TODO:
            it possible to make custom sklearn estimator through statmodels API for gs.
            deprecate all transformation here, transform target through conf.py.

        """

        if pipeline:
            steps = pipeline.steps
        else:
            steps = []

        if steps[-1][0] == 'estimate':
            del steps[-1]

        # x_train = x
        # y_train = y
        # x_test = x
        # y_test = y
        x_train, x_test, y_train, y_test = self.split(x, y)

        # will use default pipline (without `estimate`  step)
        transformer = sklearn.pipeline.Pipeline(steps, memory=None)
        # alternative: set zero position params from hp_grid
        # for name, vals in self.p['hp_grid'].items():
        #     if name.startswith('estimate'):
        #         continue
        #     transformer.set_params(**{name: vals[0]})
        x_train_ = transformer.fit_transform(x_train, y_train)
        x_test_ = transformer.transform(x_test)

        # add intercept (statmodels don`t auto add)
        x_train_ = np.c_[x_train_, np.ones(x_train_.shape[0])]
        x_test_ = np.c_[x_test_, np.ones(x_test_.shape[0])]

        fitted_estimator = self.fit_sm(x_train_, y_train)
        y_pred = self.predict(fitted_estimator, x_test_, y_test)
        self.score(y_test, y_pred)

        if self.p['pipeline__type'] == 'regressor':
            fitted_estimator, y_train_inv, transformer = self.target_normalization(fitted_estimator, x_train_, y_train)
            if transformer:
                y_test_inv = transformer.transform(y_test.reshape(len(y_test), 1)).reshape(len(y_test))
                # y_test = scipy.stats.yeojohnson(y_test, lmbda=lmbda)
                y_pred_inv = self.predict(fitted_estimator, x_test_, y_test_inv)
                y_pred = transformer.inverse_transform(y_pred_inv.reshape(len(y_pred_inv), 1)).reshape(len(y_pred_inv))
                self.score(y_test, y_pred)

        fitted_estimator, cov_type = self.handle_homo(fitted_estimator, x_train, y_train)
        # why not work with cov_type?
        # self.predict(fitted_estimator, x_test_, y_test)
        self.check_leverage(fitted_estimator)

        # [deprecated] add/drop intercept
        # self.data_df=self.data_df.assign(**{'feature_intercept':np.full(self.data.shape[0],
        #     fill_value=1, dtype=np.float64, order='C')})
        # self.data['feature_intercept'] = 1
        # delete intercept from data not to influence forward analyse
        # self.data_df.drop(['feature_intercept'], axis=1, inplace=True)

    def split(self, x, y):
        """Split data on train, test

        Note:
            if `split_train_size` set to 1.0, use full dataset to CV (test=train)

        """
        if self.p['data__split_train_size'] == 1.0:
            x_train = x_test = x
            y_train = y_test = y
        else:
            x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
                x, y,
                train_size=self.p['data__split_train_size'], test_size=None,
                shuffle=False, stratify=None)
        return x_train, x_test, y_train, y_test

    def fit_sm(self, x, y, cov_type='nonrobust'):
        """Fit statmodel"""
        x_train = x
        y_train = y
        test = None
        if self.p['pipeline__type'] == 'regressor':
            estimator = sm.OLS(y, x)
        else:
            estimator = sm.Logit(y, x)
        estimator = estimator.fit(cov_type=cov_type)
        self.logger.info('{}'.format(estimator.summary()))

        return estimator

    def predict(self, estimator, x, y):
        # work for both, sklearn and statmodels.
        y_pred = estimator.predict(x)
        return y_pred

    def score(self, y_true, y_pred):
        if self.p['pipeline__type'] == 'regressor':
            score = sklearn.metrics.mean_squared_error(y_true, y_pred, squared=False)
            self.logger.info(f"RMSE score = {score}")
        else:
            score = sklearn.metrics.log_loss(y_true, y_pred)
            self.logger.info(f"Log loss score = {score}")

    def target_normalization(self, estimator, x, y, cov_type='nonrobust'):
        """ Check normality for residuals, y yeojohnson transform if necessary"""
        _, p = scipy.stats.shapiro(estimator.resid)
        self.logger.info('Shapiro criteria p-value={}'.format(p))
        transformer = None
        if p < 0.05:
            self.logger.warning('Warning: use Yeo-Johnson normalization for target')
            # skew = self.estimator.diagn['skew']
            # kurtosis = self.estimator.diagn['kurtosis']

            # yeojohnson transform, box-cox only for positive, scipy differ from sklearn
            # y, lmbda = scipy.stats.yeojohnson(y)  # with copy
            transformer = sklearn.preprocessing.PowerTransformer(method='yeo-johnson', standardize=True, copy=True)
            # lmbda = transformer.lambdas_[0]
            y_inv = transformer.fit_transform(y.reshape(len(y), 1)).reshape(len(y))

            # refit model
            estimator = self.fit_sm(x, y_inv)
            _, p_new = scipy.stats.shapiro(estimator.resid)
            self.logger.info('Shapiro criteria after target normalization p_value = {}'.format(p_new))
        if self.plot_flag:
            # visual check Q-Q for residuals
            plt.figure(figsize=(16, 7))
            plt.subplot(121)
            scipy.stats.probplot(estimator.resid, dist="norm", plot=plt)
            plt.subplot(122)
            plt.hist(np.log(estimator.resid))
            plt.xlabel('Residuals', fontsize=14)
            plt.show()
        return estimator, y_inv, transformer

    def handle_homo(self, estimator, x, y, cov_type='nonrobust'):
        """Check homo"""
        p = sms.het_breuschpagan(estimator.resid, estimator.model.exog)[1]
        self.logger.info('Breusch-Pagan criteria p-value={}'.format(p))
        if p < 0.05:
            self.logger.warning('Warning:  apply HC1'.format(p))
            cov_type = 'HC1'
            estimator = self.fit_sm(x, y, cov_type=cov_type)
        return estimator, cov_type

    def check_leverage(self, estimator):
        """Check leverage"""
        if self.plot_flag:
            fig, ax = plt.subplots()
            fig.set_size_inches(8, 7)
            smg.plot_leverage_resid2(estimator, ax=ax)
        self.logger.debug('UNRECONSTRUCTED leverage')

    def df_info(self, data_df):
        """Log info() and describe() for data_df"""
        buf = io.StringIO()
        data_df.info(buf=buf)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            self.logger.info('{}\n'.format(tabulate.tabulate(data_df.describe(), headers='keys', tablefmt='psql')))
            self.logger.info('{}\n'.format(buf.getvalue()))
            # self.logger.info('Memory usage{}\n'.format(data_df.memory_usage())) by column
        buf.close()

    def check_imbalance(self, data_df):
        """Check and log high imbalanced features, targets"""
        # if categorical or count_values < 10
        categoric_names = [i[0] for i in self.categoric_ind_name.values()]
        numeric_names = [i[0] for i in self.numeric_ind_name.values()]
        # calc unique values (np.nan not included as value)
        value_counts = {column_name: data_df[column_name].value_counts()
                        for i, column_name in enumerate(data_df.columns)}

        if self.p['pipeline__type'] == 'classifier':
            categoric_names += ['targets']
        else:
            numeric_names += ['targets']
        for column_name in categoric_names + numeric_names:
            column = value_counts[column_name]
            if column_name in numeric_names and column.size > 10:
                continue
            amounts = column.values  # amount
            uniq_values = column.index   # value
            for i, val in enumerate(amounts):
                if float(val) / column.size < 0.05:
                    self.logger.warning('Warning: class imbalanced: {} {:.2f}<5% {}'
                                        .format(int(uniq_values[i]), 100 * float(val) / column.sum(), column_name))
                    # [deprecated] class balance for all data
                    # self.logger.debug('{}\n'.join(map(lambda x:'{}\n{}'
                    # .format(x[0],x[1]),[(column_name, value_counts[column_name])
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

    def plottings(self, data_df, indices=None):
        """Plot scatter-matrix for numeric columns in dataframe

        Args:
            data_df
            indices (list): column names

        """
        # scatter-plot for continuous data
        if indices is None:
            indices = self.numeric_ind_name

        if 0 < len(indices) < 15:
            pd.plotting.scatter_matrix(self.subcolumns(data_df, indices=indices),
                                       alpha=0.2, figsize=(15, 15), diagonal='hist')
            plt.show()
        else:
            self.logger.warning('Warning: Numeric columns for plot scatter matrix'
                                ' should be in (0,15), {} given'.format(len(indices)))

    def subcolumns(self, x, **kwargs):
        """Get subcolumns from x.

        Args:
            x (np.ndarray or dataframe of shape=[[row],]): Input x.
            **kwargs: Should contain 'indices' key.

        Returns:
            result (np.ndarray or xframe): Subcolumns of x.
        """
        feat_ind_name = kwargs['indices']
        indices = list(feat_ind_name.keys())
        names = [i[0] for i in feat_ind_name.values()]
        if isinstance(x, pd.DataFrame):
            return x.loc[:, names]
        else:
            return x[:, indices]


if __name__ == '__main__':
    eda = mlshell.EDA(project_path, logger=logger, params=params)
    eda.analyze_data(wf.data_df, pipeline=wf.estimator,
                     categoric_ind_name=wf.categoric_ind_name,
                     numeric_ind_name=wf.numeric_ind_name,
                     plot_flag=False)
