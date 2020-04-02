from mlshell.libs import *


class GetData(object):
    def __init__(self, logger=None):
        if logger is None:
            self.logger = logging.Logger('GetData')
        else:
            self.logger = logger
        self.raw = None  # data attribute, fullfil in self.get_data()

    # @memory_profiler
    def get_data(self, filename1, filename2, **kwargs):
        self.logger.info("\u25CF LOAD DATA")
        transaction = self.get_csv(filename1, **kwargs)
        identity = self.get_csv(filename2, **kwargs)
        self.raw = pd.merge(transaction, identity,
                            left_index=True, right_index=True,
                            how='left', suffixes=('_left', '_right'))
        self.logger.info("Data loaded from:\n    {}".format('\n    '.join([filename1, filename2])))
        return self.raw

    def get_csv(self, filename, rows_limit=None, random_skip=False, index_col=None):
        """ Get data from csv-file.

        Args:
            filename (str): Relative path to csv file with data.
            rows_limit (int or None, optional (default=None)): Number of lines get from input file.
            random_skip (bool, optional (default=False)): If True and rows_limit=True get rows random from input.
            index_column (str, optional (default=None)): Index column name in .csv file.

        Notes:
            skiprows index strat from 0
                default None
            nrows is working with connection to skiprows
                default None
            it is much more faster than read full anyway
            headers shoukd be otherwise return rows_limit+1

        """
        currentpath = os.path.dirname(os.path.abspath(__file__))
        filename = "{}/{}".format(currentpath, filename)
        # count lines
        with open(filename, 'r') as f:
            lines = sum(1 for _ in f)
        if rows_limit:
            if rows_limit > lines:
                rows_limit = None
                skip_list = None
            elif random_skip:
                skip_list = rd.sample(range(1, lines), lines - rows_limit - 1)
            else:
                skip_list = None
        else:
            skip_list = None

        with open(filename, 'r') as f:
            raw = pd.read_csv(f, sep=",", index_col=index_col, skiprows=skip_list, nrows=rows_limit)

        return raw


class DataPreprocessor(object):
    # @time_profiler
    # @memory_profiler
    def __init__(self, logger=None):
        if logger is None:
            self.logger = logging.Logger('DataPreprocessor')
        else:
            self.logger = logger
        self.logger = logger
        self.data = None
        self.raw_names = None
        self.base_plot = None
        self.raw_index_names = None
        self.raw_targets_names = None
        self.raw_features_names = None

    def preprocess_data(self, raw):
        self.logger.info("\u25CF PREPROCESS DATA")
        target_name = 'isFraud'
        categor_names = ['ProductCD', 'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain'] + \
                        [f'M{i}' for i in range(1, 10)] + \
                        [f'card{i}' for i in range(1, 7)] + \
                        ['DeviceType', 'DeviceInfo'] + \
                        [f'id_{i}' for i in range(12, 39)]
        index = raw.index
        targets, self.raw_targets_names, self.base_plot = self.make_targets(raw, target_name=target_name)
        features, self.raw_features_names = self.make_features(raw, target_name=target_name)
        self.data = self.make_dataframe(index, targets, features, categor_names)
        self.raw_names = {'index': self.raw_index_names,
                          'targets': self.raw_targets_names,
                          'features': self.raw_features_names}
        return self.data, self.raw_names, self.base_plot

    def make_targets(self, raw, target_name=''):
        try:
            targets_df = raw[target_name]
            targets = targets_df.values.astype(int)  # cast to int
        except KeyError as e:
            # handle test data without targets
            self.logger.warning("Warning: no target column '{}' in df, use 0 values.".format(target_name))
            targets = np.zeros(raw.shape[0], dtype=int, order="C")
            raw[target_name] = targets
        raw_targets_names = [target_name]
        # base_plot = targets
        # preserve original index
        # base_plot = pd.Series(index=raw.index.values,
        #                       data=np.arange(1, targets.shape[0]+1)).rename_axis(raw.index.name)

        base_plot = pd.DataFrame(index=raw.index.values,
                                 data={target_name: np.arange(1, targets.shape[0]+1)}).rename_axis(raw.index.name)

        return targets, raw_targets_names, base_plot

    def make_features(self, raw, target_name=''):
        features_df = raw.drop([target_name], axis=1)
        # there is mistake with '-' test column names in input data
        raw_features_names = [i.replace('-', '_') for i in features_df.columns]
        features = features_df.values.T
        return features, raw_features_names

    def make_dataframe(self, index, targets, features, categor_names):
        columns = [f'feature_categor_{i}__{raw_name}' if raw_name in categor_names
                   else f'feature_{i}__{raw_name}'
                   for i, raw_name in enumerate(self.raw_features_names)]
        df = pd.DataFrame(
            data=features.T,
            index=index,
            columns=columns,
            copy=False,
        )
        df.insert(loc=0, column='targets', value=targets)
        return df


if __name__ == '__main__':
    pass
