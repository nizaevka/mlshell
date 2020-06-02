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
        # [deprecated] self.raw = None  # data attribute, fullfill in self.get_data()

    # @memory_profiler
    def get(self, dataset, filename=None, rows_limit=None, random_skip=False, index_col=None, **kwargs):
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
                random_state = sklearn.utils.check_random_state(kwargs.get('random_state', None))
                skip_list = random_state.choice(range(1, lines), size=lines - rows_limit - 1, replace=False, p=None)
                # [deprecated] bad practice to use global seed
                # rd.seed(42)
                # skip_list = rd.sample(range(1, lines), lines - rows_limit - 1)
            else:
                skip_list = None
        else:
            skip_list = None
        with open(filename, 'r') as f:
            raw = pd.read_csv(f, sep=",", index_col=index_col, skiprows=skip_list, nrows=rows_limit)
        self.logger.info("Data loaded from:\n    {}".format(filename))

        dataset['raw'] = raw
        return dataset


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

    def preprocess(self, dataset, target_names=None, categor_names=None, pos_labels=None, **kwargs):
        self.logger.info("\u25CF \u25B6 PREPROCESS DATA")
        raw = dataset['raw']
        if categor_names is None:
            categor_names = []
        if target_names is None:
            target_names = []
        if pos_labels is None:
            pos_labels = []
        index = raw.index
        targets = self.make_targets(raw, target_names)
        features, features_names = self.make_features(raw, target_names)
        # TODO: pickle
        #     better add names postfix
        raw_names = {'index': list(index),  # otherwise not serializable for cache
                     'index_name': index.name,
                     'targets': target_names,
                     'features': list(features_names),  # otherwise not serializable for cache
                     'categor_features': categor_names,
                     'pos_labels': pos_labels}
        data = self.make_dataframe(index, targets, features, raw_names)
        dataset.update({'data': data, 'raw_names': raw_names})  # [deprecated] , 'base_plot': base_plot
        del dataset['raw']
        return dataset

    def make_targets(self, raw, target_names):
        """Targets preprocessing."""
        try:
            targets_df = raw[target_names]
            targets = targets_df.values.astype(int)  # cast to int
        except KeyError as e:
            # handle test data without targets
            self.logger.warning("Warning: no target column(s) '{}' in df, use 0 values.".format(target_names))
            targets = np.zeros((raw.shape[0], len(target_names)), dtype=int, order="C")
            raw[target_names] = pd.DataFrame(targets)
        # [deprecated] better df
        # base_plot = targets
        # preserve original index
        # base_plot = pd.Series(index=raw.index.values,
        #                       data=np.arange(1, targets.shape[0]+1)).rename_axis(raw.index.name)

        # [deprecated] move to gui
        # base_plot = pd.DataFrame(index=raw.index.values,
        #                          data={target_name: np.arange(1, targets.shape[0]+1)}).rename_axis(raw.index.name)

        return targets # , base_plot

    def make_features(self, raw, target_names):
        """Features preprocessing."""
        features_df = raw.drop(target_names, axis=1)
        raw_features_names = features_df.columns
        features = features_df.values.T
        return features, raw_features_names

    def make_dataframe(self, index, targets, features, raw_names):
        """Combine preprocessed columns."""
        # [deprecated] preserve original
        # columns = [f'feature_categor_{i}__{raw_name}' if raw_name in raw_names['categor_features']
        #            else f'feature_{i}__{raw_name}'
        #            for i, raw_name in enumerate(raw_names['features'])]
        columns = raw_names['features']
        df = pd.DataFrame(
            data=features.T,
            index=index,
            columns=columns,
            copy=False,
        ).rename_axis(raw_names['index_name'])
        df.insert(loc=0, column='targets', value=targets)
        return df


class Dataset(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __hash__(self):
        return pd.util.hash_pandas_object(self.get('data')).sum()

    # good choice, cause for train,
    # otherwise if attribute, for train, test we will inherent x,y,classes => complicated, need oo clean)
    # need for ram economy, we can make it as dict key
    # But: multiple calls
    def get_x(self):
        df = self.get('data', None)
        raw_names = self.get('raw_names', None)
        return df[raw_names['features']]

    def get_y(self):
        df = self.get('data', None)
        raw_names = self.get('raw_names', None)
        return df[raw_names['targets']]

    # TODO: remove is_classifier check if no need
    def get_classes(self, pos_labels=None):
        df = self.get('data', None)
        raw_names = self.get('raw_names', None)
        if not pos_labels:
            pos_labels = raw_names.get('pos_labels', [])

        classes = [np.unique(j) for i, j in df[raw_names['targets']].iteritems()]  # [array([1]), array([2, 7])]
        if not pos_labels:
            pos_labels_ind = -1
            pos_labels = [i[-1] for i in classes]  # [2,4]
        else:
            # Find where pos_labels in sorted labels.
            pos_labels_ind = [np.where(classes[i] == pos_labels[i])[0][0] for i in range(len(classes))]  # [1, 0]

        # [deprecated] not work if different number of classes in multi-output
        # if not pos_labels:
        #     pos_labels = classes[..., -1]  # [2,4]
        #     pos_labels_ind = -1  # [wrong] np.full(pos_labels.shape, fill_value=-1)
        # else:
        #     pos_labels_ind = np.array(np.where(classes == pos_labels))[..., 0]  # [1, 0]

        # [temp]
        print(f"Label {pos_labels} identified as positive np.unique(targets)[-1]:\n"
              f"    for classifiers provided predict_proba:"
              f" if P(pos_labels)>threshold, prediction=pos_labels on sample.")

        return {'classes': classes, 'pos_labels': pos_labels, 'pos_labels_ind': pos_labels_ind}

    def dump(self, filepath, obj, template=None):
        # [deprecated]
        # raw_names = self.get('raw_names')
        if template:
            # recover original index and names
            obj = pd.DataFrame(index=template.index.values,
                               data={zip(template.columns, obj)}).rename_axis(template.index.name)
            # [deprecated] not enough abstract
            # df = pd.DataFrame(index=self.get('data').index.values,
            #                data={raw_names['targets'][0]: y_pred}).rename_axis(raw_names['index'])

        with open(f"{filepath}.csv", 'w', newline='') as f:
            obj.to_csv(f, mode='w', header=True, index=True, sep=',', line_terminator='\n')  # only LF
        return

    def split(self):
        df = self.get('data', None)
        train_index = self.get('train_index', None)
        test_index = self.get('test_index', None)
        if train_index is None and test_index is None:
            train_index = test_index = df.index

        # inherent keys, except 'data'
        train = Dataset(dict(self, **{'data': df.loc[train_index]}))
        test = Dataset(dict(self, **{'data': df.loc[test_index]}))

        return train, test

        # [deprecated] better create subobject
        # columns = df.columns
        # # deconcatenate without copy, better dataframe over numpy (provide index)
        # train = df.loc[train_index]
        # test = df.loc[test_index]
        # x_train = train[[name for name in columns if 'feature' in name]]
        # y_train = train['targets']
        # x_test = test[[name for name in columns if 'feature' in name]]
        # y_test = test['targets']
        # return (x_train, y_train), (x_test, y_test)


class DataFactory(DataExtractor, DataPreprocessor):
    def __init__(self, project_path, logger=None):
        super().__init__(project_path, logger)
        if logger is None:
            self.logger = logging.Logger('DataFactory')
        else:
            self.logger = logger
        self.project_path = project_path

    def produce(self, dataset_id, p):
        """ Read dataset and Unify dataframe in compliance to workflow class.

        Arg:
            p():
        Note:
            Else: run unifer without cahing results.

        """
        self.logger.info("\u25CF HANDLE DATA")
        self.logger.info(f"Data configuration:\n    {dataset_id}")
        res_ = Dataset()
        for key, val in p.get('steps', {}):
            if not isinstance(val, dict):
                continue
            # [deprecated] conf comments better than flag
            # if key == 'dump_cache' and not val['flag']:
            #     continue
            if 'prefix' in val and not val['prefix']:
                val['prefix'] = dataset_id
            res_ = getattr(self, key)(res_, **val)
            # [deprecated] conf comments better than flag, also contain error dataser={}
            # if key == 'load_cache' and val['flag'] and res_:
            if key == 'load_cache' and res_:
                break

        res = self.check(res_)


        # [deprecated]
        # # cache flag False, True, update
        # if self.p['cache'] and not self.p['cache'] == 'update':
        #     cache = self.load_cache(prefix=dataset_id)
        #     # [deprecated] now cache arbitrary types
        #     # cache, meta = self.load_cache(prefix=dataset_id)
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
        #         self.dump_cache(res, prefix=dataset_id)
        # else:
        #     res = cache

        return res

    def dump_cache(self, dataset, prefix='', **kwargs):
        """Dump intermediate dataframe to disk."""
        # TODO: pickle could be more universal (better dill).
        cachedir = f"{self.project_path}/results/cache/data"
        if not os.path.exists(cachedir):
            # create temp dir for cache if not exist
            os.makedirs(cachedir)
        for filename in glob.glob(f"{cachedir}/{prefix}*"):
            os.remove(filename)
        fps = set()
        for key, val in dataset.items():
            if isinstance(dataset[key], (pd.DataFrame, pd.Series)):
                filepath = f'{cachedir}/{prefix}_{key}_.csv'
                fps.add(filepath)
                with open(filepath, 'w', newline='') as f:
                    val.to_csv(f, mode='w', header=True, index=True, line_terminator='\n')
            elif isinstance(dataset[key], np.ndarray):
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
        return dataset

    def load_cache(self, dataset, prefix='', **kwargs):
        """Load intermediate dataframe from disk"""
        cachedir = f"{self.project_path}/results/cache/data"
        dataset = Dataset()
        for filepath in glob.glob(f"{cachedir}/{prefix}*"):
            key = '_'.join(filepath.split('_')[1:-1])
            if filepath.endswith('.csv'):
                with open(filepath, 'r') as f:
                    dataset[key] = pd.read_csv(f, sep=",", index_col=0)
            else:
                with open(filepath, 'r') as f:
                    dataset[key] = dict(json.load(f))  # object_hook=json_keys2int)
        self.logger.warning(f"Warning: use cache file(s):\n    {cachedir}")
        return dataset

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

    def info(self, dataset, **kwargs):
        self.check_duplicates(dataset['data'])
        self.check_gaps(dataset['data'])
        return dataset

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

    def unify(self, dataset, **kwargs):
        """ unify input dataframe

        Note:
            * delete duplicates, (not reset index, otherwise problem with base_plot).
            * log info about gaps.
            * unify gaps.

                * if gap in targets => raise MyException
                * if gap in categor => 'unknown'(downcast dtype to str) => ordinalencoder
                * if gap in non-categor => np.nan
            * transform to np.float64 (python float = np.float = np.float64 = C double = np.double(64 bit processor)).
            * define dictionaries of indices (when drop targets):

                * self.categoric_ind_name => {1:('feat_n', ['cat1', 'cat2'])}
                * self.numeric_ind_name   => {2:('feat_n',)}

        Returns:
            data (pd.DataFrame): unified input dataframe
            categoric_ind_name (dict): {column_index: ('feature_categr__name',['B','A','C']),}
            numeric_ind_name (dict):  {column_index: ('feature__name',),}

        """
        data = dataset['data']
        raw_names = dataset['raw_names']
        categoric_ind_name = {}
        numeric_ind_name = {}
        count = 0
        for ind, column_name in enumerate(data):
            if column_name in raw_names['targets']:
                count += 1
                continue
            if column_name in raw_names['categor_features']:
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
                categoric_ind_name[ind-count] = (column_name,
                                                 encoder.categories_[0].tolist())
            else:
                # fill gaps with np.nan
                data[column_name].fillna(value=np.nan, method=None, axis=None,
                                         inplace=True, limit=None, downcast=None)
                numeric_ind_name[ind-count] = (column_name,)
        # cast to np.float64 without copy
        # alternative: try .to_numeric

        # TODO: try built-in alternative
        #    sklearn.utils.as_float_array
        #    assert_all_finite
        #    https://scikit-learn.org/stable/developers/utilities.html#developers-utils
        data = data.astype(np.float64, copy=False, errors='ignore')
        dataset.update({'data': data, 'categoric_ind_name': categoric_ind_name, 'numeric_ind_name': numeric_ind_name})
        return dataset

    # @memory_profiler
    def split(self, dataset, **kwargs):
        """Split data on train, test

        data (pandas.DataFrame, optional (default=None)):
            if not None ``dataset_id`` ignored, read kwargs.
        dataset_id (str, optional (default='train')):
            | should be known key from params['data`]
            | if None, used default ``dataset_id`` from params['fit__dataset_id'] and corresponding kwargs.
        kwargs:
            if dataset_id is not None, ignore current, use global from params['data__dataset_id__split'].

        Note:
            input data updated inplace with additional split key.
            if split ``train_size`` set to 1.0, use test=train.
        """
        self.logger.info("\u25CF SPLIT DATA")
        data = dataset['data']

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
        dataset.update({'train_index': train_index, 'test_index': test_index})
        return dataset

    def check(self, dataset, **kwargs):
        dataset = self._check_numeric_types(dataset, **kwargs)
        return dataset

    def _check_numeric_types(self, dataset, **kwargs):
        # check that all non-categoric features are numeric type
        data = dataset['data']
        dtypes = data.dtypes
        misstype = []
        for ind, column_name in enumerate(data):
            if '_categor_' not in column_name:
                if not np.issubdtype(dtypes[column_name], np.number):
                    misstype.append(column_name)
        if misstype:
            raise ValueError("Input data non-categoric columns"
                             " should be subtype of np.number, check:\n    {}".format(misstype))
        return dataset


if __name__ == '__main__':
    pass
