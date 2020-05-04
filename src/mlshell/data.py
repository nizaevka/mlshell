""""
data = class
    .split() should return train, test
    .df
Use this as identifiers for hp_grid
Maybe in sklearn there are special format for datasets?

"""

from mlshell.libs import *


class DataExtractor(object):
    def __init__(self, project_path, logger=None):
        if logger is None:
            self.logger = logging.Logger('GetData')
        else:
            self.logger = logger
        self.project_path = project_path
        # self.raw = None  # data attribute, fullfill in self.get_data()

    # @memory_profiler
    def get(self, var, filename=None, rows_limit=None, random_skip=False, index_col=None, **kwargs):
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
        self.logger.info("\u25CF \u25B6 LOAD DATA")
        # [deprecated]
        # currentpath = os.path.dirname(os.path.abspath(__file__))
        filename = "{}/{}".format(self.project_path, filename)
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
        self.logger.info("Data loaded from:\n    {}".format(filename))

        var['raw'] = raw
        return var


class DataPreprocessor(object):
    # @time_profiler
    # @memory_profiler
    def __init__(self, project_path, logger=None):
        super().__init__(logger)
        if logger is None:
            self.logger = logging.Logger('DataPreprocessor')
        else:
            self.logger = logger
        self.logger = logger
        self.project_path = project_path

    def preprocess(self, var, target_name='', categor_names=None, **kwargs):
        self.logger.info("\u25CF \u25B6 PREPROCESS DATA")
        raw = var['raw']
        if categor_names is None:
            categor_names = []
        index = list(raw.index)  # otherwise not serializable for cache
        targets, raw_targets_names, base_plot = self.make_targets(raw, target_name=target_name)
        features, raw_features_names = self.make_features(raw, target_name=target_name)
        data = self.make_dataframe(index, targets, features, categor_names, raw_features_names)
        raw_names = {'index': index,
                     'targets': raw_targets_names,
                     'features': raw_features_names}
        var.update({'df': data, 'raw_names': raw_names, 'base_plot': base_plot})
        del var['raw']
        return var

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

    def make_dataframe(self, index, targets, features, categor_names, raw_features_names):
        columns = [f'feature_categor_{i}__{raw_name}' if raw_name in categor_names
                   else f'feature_{i}__{raw_name}'
                   for i, raw_name in enumerate(raw_features_names)]
        df = pd.DataFrame(
            data=features.T,
            index=index,
            columns=columns,
            copy=False,
        )
        df.insert(loc=0, column='targets', value=targets)
        return df


class DataFactory(DataExtractor, DataPreprocessor):
    def __init__(self, project_path, logger=None):
        super().__init__(project_path, logger)
        if logger is None:
            self.logger = logging.Logger('DataFactory')
        else:
            self.logger = logger
        self.project_path = project_path

    def produce(self, data_id, p):
        """ Read dataset and Unify dataframe in compliance to workflow class.

        Arg:
            p():
        Note:
            Else: run unifer without cahing results.

        """
        self.logger.info("\u25CF HANDLE DATA")
        self.logger.info(f"Data configuration:\n    {data_id}")
        res_ = {}
        for key, val in p.items():
            if not isinstance(val, dict):
                continue
            if key == 'dump_cache' and not val['flag']:
                continue
            if 'prefix' in val and not val['prefix']:
                val['prefix'] = data_id
            res_ = getattr(self, key)(res_, **val)
            if key == 'load_cache' and val['flag'] and res_:
                break

        res = self.check(res_)

        # [deprecated]
        # # cache flag False, True, update
        # if self.p['cache'] and not self.p['cache'] == 'update':
        #     cache = self.load_cache(prefix=data_id)
        #     # [deprecated] now cache arbitrary types
        #     # cache, meta = self.load_cache(prefix=data_id)
        #     # if cache is not None:
        #     #     data = cache
        #     #     data_df = data
        #     #     categoric_ind_name = meta['categoric']
        #     #     numeric_ind_name = meta['numeric']
        # else:
        #     cache = None

        # if cache is None:
        #     res_ = None
        #     for key, val in p.items():
        #         if not isinstance(val, dict):
        #             continue
        #         res_ = getattr(self, key)(res_, **val)
        #     res = res_

        #     # [deprecated] manual
        #     # raw = self.get(**p['get'])
        #     # data = self.preprocess(raw, **p['preprocess'])
        #     # data = self.check(data, **p['check'])
        #     # data = self.unify(data, **p['unify'])

        #     # [deprecated] move to workflow
        #     # self.check_numeric_types(data_df)
        #     if self.p['cache']:
        #         self.dump_cache(res, prefix=data_id)
        # else:
        #     res = cache

        return res

    def dump_cache(self, var, prefix='', **kwargs):
        """Dump imtermediate dataframe to disk."""
        cachedir = f"{self.project_path}/results/cache/data"
        if not os.path.exists(cachedir):
            # create temp dir for cache if not exist
            os.makedirs(cachedir)
        for filename in glob.glob(f"{cachedir}/{prefix}*"):
            os.remove(filename)
        fps = set()
        for key, val in var.items():
            if isinstance(var[key], (pd.DataFrame, pd.Series)):
                filepath = f'{cachedir}/{prefix}_{key}_.csv'
                fps.add(filepath)
                with open(filepath, 'w', newline='') as f:
                    val.to_csv(f, mode='w', header=True, index=True, line_terminator='\n')
            elif isinstance(var[key], np.ndarray):
                filepath = f'{cachedir}/{prefix}_{key}_.csv'
                fps.add(filepath)
                with open(filepath, 'w', newline='') as f:
                    pd.DataFrame(val).to_csv(f, mode='w', header=True, index=True, line_terminator='\n')
                # np.savetxt(filepath, val, delimiter=",")
            else:
                filepath = f'{cachedir}/{prefix}_{key}_.json'
                fps.add(filepath)
                with open(filepath, 'w') as f:
                    # items() preserve first level dic keys as int
                    json.dump(list(val.items()), f)

        self.logger.warning('Warning: update cache file(s):\n    {}'.format('\n    '.join(fps)))
        return var

    def load_cache(self, var, prefix='', **kwargs):
        """Load intermediate dataframe from disk"""
        cachedir = f"{self.project_path}/results/cache/data"
        var = {}
        for filepath in glob.glob(f"{cachedir}/{prefix}*"):
            key = filepath.split('_')[1]
            if filepath.endswith('.csv'):
                with open(filepath, 'r') as f:
                    var[key] = pd.read_csv(f, sep=",", index_col=0)
            else:
                with open(filepath, 'r') as f:
                    var[key] = dict(json.load(f))  # object_hook=json_keys2int)
        self.logger.warning(f"Warning: use cache file(s):\n    {cachedir}")
        return var

    # def dump_cache(self, data, categoric_ind_name, numeric_ind_name, prefix=''):
    #     """Dump imtermediate dataframe to disk."""
    #     cachedir = f"{self.project_path}/results/cache/unifier"
    #     filepath = f'{cachedir}/{prefix}_after_unifier.csv'
    #     filepath_meta = f'{cachedir}/{prefix}_after_unifier_meta.json'
    #     if self.p['cache']:
    #         if self.p['cache'] == 'update':
    #             if os.path.exists(filepath):
    #                 os.remove(filepath)
    #             if os.path.exists(filepath):
    #                 os.remove(filepath_meta)
    #             # shutil.rmtree(cachedir, ignore_errors=True)
    #         if not os.path.exists(cachedir):
    #             # create temp dir for cache if not exist
    #             os.makedirs(cachedir)
    #         # only if cache is None
    #         self.logger.warning('Warning: update unifier cache file:\n    {}'.format(filepath))
    #         with open(filepath, 'w', newline='') as f:
    #             data.to_csv(f, mode='w', header=True, index=True, line_terminator='\n')
    #         column_ind_name = {'categoric': categoric_ind_name, 'numeric': numeric_ind_name}
    #         with open(filepath_meta, 'w') as f:
    #             json.dump(column_ind_name, f)

    # def load_cache(self, prefix=''):
    #     """Load intermediate dataframe from disk"""
    #     cachedir = f"{self.project_path}/results/cache/unifier"
    #     filepath = f'{cachedir}/{prefix}_after_unifier.csv'
    #     filepath_meta = f'{cachedir}/{prefix}_after_unifier_meta.json'
    #     if self.p['cache__unifier'] \
    #             and os.path.exists(filepath) \
    #             and os.path.exists(filepath_meta) \
    #             and not self.p['cache__unifier'] == 'update':
    #         with open(filepath, 'r') as f:
    #             cache = pd.read_csv(f, sep=",", index_col=0)
    #         with open(filepath_meta, 'r') as f:
    #             meta = json.load(f, object_hook=json_keys2int)
    #         self.logger.warning(f"Warning: use cache file instead unifier:\n    {cachedir}")
    #         return cache, meta
    #     return None, None

    def info(self, var, **kwargs):
        self.check_duplicates(var['df'])
        self.check_gaps(var['df'])
        return var

    def check_duplicates(self, data, del_duplicates=None):
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

    def unify(self, var, **kwargs):
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
        data = var['df']
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
        var.update({'df': data, 'categoric_ind_name': categoric_ind_name, 'numeric_ind_name': numeric_ind_name})
        return var

    # @memory_profiler
    def split(self, var, **kwargs):
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
        data = var['df']

        if (kwargs['train_size'] == 1.0 and kwargs['test_size'] is None
            or kwargs['train_size'] is None and kwargs['test_size'] == 0):
            train = test = data
            train_index, test_index = data.index
        else:
            shell_kw = ['func']
            kwargs = copy.deepcopy(kwargs)
            for kw in shell_kw:
                kwargs.pop(kw)
            train, test, train_index, test_index = sklearn.model_selection.train_test_split(
                data, data.index.values, **kwargs)

        # add to data
        var.update({'train_index': train_index, 'test_index': test_index})
        return var

    def check(self, var, **kwargs):
        var = self._check_numeric_types(var, **kwargs)
        return var

    def _check_numeric_types(self, var, **kwargs):
        # check that all non-categoric features are numeric type
        data = var['df']
        dtypes = data.dtypes
        misstype = []
        for ind, column_name in enumerate(data):
            if '_categor_' not in column_name:
                if not np.issubdtype(dtypes[column_name], np.number):
                    misstype.append(column_name)
        if misstype:
            raise ValueError("Input data non-categoric columns"
                             " should be subtype of np.number, check:\n    {}".format(misstype))
        return var


if __name__ == '__main__':
    pass
