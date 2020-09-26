from mlshell.libs import *


class GetData(object):
    def __init__(self, logger=None):
        if logger is None:
            self.logger = logging.Logger('GetData')
        else:
            self.logger = logger
        self.raw = None  # data attribute, filled in self.get_data()

    # @memory_profiler
    def get_data(self, filename, rows_limit=None, random_skip=False, index_col=None):
        """ Get data from csv-file.

        Args:
            filename (str): Relative path to csv file with data.
            rows_limit (int or None, optional (default=None)): Number of lines get from input file.
            random_skip (bool, optional (default=False)): If True and rows_limit=True get rows random from input.
            index_col (str, optional (default=None)): Index column name in .csv file.

        Notes:
            skiprows index strat from 0
                default None
            nrows is working with connection to skiprows
                default None
            it is much more faster than read full anyway
            headers shoukd be otherwise return rows_limit+1

        """
        self.logger.info("\u25CF LOAD DATA")
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
            self.raw = pd.read_csv(f, sep=",", index_col=index_col, skiprows=skip_list, nrows=rows_limit)
        self.logger.info("Data loaded from:\n    {}".format(filename))
        return self.raw


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
        target_name = 'loss'
        categor_names = [i for i in raw.columns if 'cat' in i]
        index = raw.index
        self.raw_index_names = raw.index.name
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
            targets = targets_df.values
        except KeyError as e:
            self.logger.warning("Warning: no target column '{}' in df, use 0 values.".format(target_name))
            targets = [0] * raw.shape[0]
            raw[target_name] = targets
        raw_targets_names = [target_name]
        # base_plot = targets
        # preserve original index
        base_plot = pd.DataFrame(index=raw.index.values,
                                 data={target_name: targets}).rename_axis(raw.index.name).sort_values(by=[target_name])
        return targets, raw_targets_names, base_plot

    def make_features(self, raw, target_name=''):
        features_df = raw.drop([target_name], axis=1)
        raw_features_names = features_df.columns
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
