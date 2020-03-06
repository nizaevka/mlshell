from mlshell.libs import *


class GetData(object):
    def __init__(self, logger):
        self.logger = logger
        self.raw = None  # data attribute, fullfil in self.get_data()

    # @memory_profiler
    def get_data(self, filename, rows_limit=None, random_skip=False):
        """ get data from csv-file

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
            self.raw = pd.read_csv(f, sep=",", index_col='id', skiprows=skip_list, nrows=rows_limit)


class DataPreprocessor(object):
    # @time_profiler
    # @memory_profiler
    def __init__(self, logger, raw):
        self.logger = logger
        index = raw.index
        self.raw_index_names = raw.index.name
        targets, self.raw_targets_names, self.base_plot = self.make_targets(raw, name='loss')
        features, self.raw_features_names = self.make_features(raw)
        self.data = self.make_dataframe(index, targets, features)

    def make_targets(self, raw, name=''):
        try:
            targets_df = raw[name]
            targets = targets_df.values
        except Exception as e:
            self.logger.warning("MyWarning: no target column '{}' in df, use 0 values/n{}".format(name, e))
            targets = [0] * raw.shape[0]
            raw[name] = targets
        raw_targets_names = [name]
        base_plot = targets
        return targets, raw_targets_names, base_plot

    def make_features(self, raw):
        features_df = raw.drop(['loss'], axis=1)
        raw_features_names = features_df.columns
        features = features_df.values.T
        return features, raw_features_names

    def make_dataframe(self, index, targets, features):
        df = pd.DataFrame(
            data=features.T,
            index=index,
            columns=[
                'feature_{}{}__{}'
                .format('categor_' if 'cat' in self.raw_features_names[i]
                        else '', i, self.raw_features_names[i]) for i in range(features.shape[0])
            ],
            copy=False,
        )
        df.insert(loc=0, column='targets', value=targets)
        return df


if __name__ == '__main__':
    pass
