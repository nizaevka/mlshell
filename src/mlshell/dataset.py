""""
The :mod:`mlshell.dataset` contains examples for `Dataset` class to create
template and `DataProducer` class to fulfills it.

For convenience `DataProducer` methods divided on:
* `DataIO` class to define IO related methods.
Implements reading from csv-file, also methods to pickle/unpickle dataset
after/before preprocessing.
* `DataPreprocessor` class for preprocessing data to final state.
Implements data transformation in compliance to `Dataset` and common exploration
techniques.

`Dataset` class intended to be used in `mlshell.Workflow`.
All data-specific methods put there, so no need to edit `Workflow` class for new
data format, only `Dataset` endpoints realization.

TODO: check what await for.
To use in Workfloe:
get_x
get_y
get_classes
dump
split
...

See also
--------
:class:`mlshell.Workflow` docstring for dataset prerequisites.

"""

#[deprecated] good choice
# alternative if attribute for train/test we will inherent x,y,classes => complicated, need oo clean)
# need for ram economy, we can make it as dict key
# But: multiple calls


from mlshell.libs import *
import mlshell


class Dataset(dict):
    """Unified data class.

    Provides inteface to work with arbitrary data.
    Interface: get_x, get_y, get_classes, dump, split

    In implementation self
    {
        'data': data,
        'raw_names':
    }

    Parameters
    ----------
    *args
    **kwrags

    """
    _required_parameters = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __hash__(self):
        """Hash value of dataset."""
        return pd.util.hash_pandas_object(self['data']).sum()

    def get_x(self):
        """Extract features from dataset.

        Keys
        ----
        data : pandas.DataFrame
            Data to split.
        raw_names : dict {'features' : ['column_1', columns_2', ..]}
            Dictionary with features column names.

        Returns
        -------
        features : pd.DataFrame
            Extracted features columns.

        """
        df = self['data']
        raw_names = self['raw_names']
        return df[raw_names['features']]

    def get_y(self):
        """Extract targets from dataset.

        Keys
        ----
        data : pandas.DataFrame
            Data to split.
        raw_names : dict {'targets' : ['columns_1', columns_2', ..]}
            Dictionary with targets column names.

        Returns
        -------
        targets : pd.DataFrame
            Extracted target columns.

        """
        df = self['data']
        raw_names = self['raw_names']
        return df[raw_names['targets']]

    def get_classes(self):
        """Extract classes and pos label index from classification dataset.

        Keys
        ----
        pos_labels : list, optional (default=None)
            Todo
            if None, last label in np.unique(target) for each target is used.

        Returns
        -------
        result : dict
            {'classes': classes,
             'pos_labels': pos_labels,
             'pos_labels_ind': pos_labels_ind}
             TODO:

        """
        df = ['data']
        raw_names = self['raw_names']
        pos_labels = raw_names.get('pos_labels', [])

        # Find classes, example: [array([1]), array([2, 7])].
        classes = [np.unique(j) for i, j in
                   df[raw_names['targets']].iteritems()]
        if not pos_labels:
            pos_labels_ind = -1
            pos_labels = [i[-1] for i in classes]  # [2,4]
        else:
            # Find where pos_labels in sorted labels, example: [1, 0].
            pos_labels_ind = [np.where(classes[i] == pos_labels[i])[0][0]
                              for i in range(len(classes))]

        print(f"Labels {pos_labels} identified as positive:\n"
              f"    for classifiers supported predict_proba:"
              f" if P(pos_labels)>threshold, prediction=pos_labels on sample.")

        return {'classes': classes,
                'pos_labels': pos_labels,
                'pos_labels_ind': pos_labels_ind}

    def split(self):
        """Split dataset on train and test.

        Keys
        ----
        data : pandas.DataFrame
            Data to split.
        train_index : array-like
            Train indices in data.
        test_index : array-like
            Test indices in data.

        Returns
        -------
        train : Dataset
            Train dataset. Inherit input dataset keys, except `data`.
        test : Dataset
            Test dataset. Inherit input dataset keys, except `data`.

        Notes
        -----
        If train_index is None/absent and test_index is None/absent:
        train=test=whole dataset.

        """
        df = self.get('data', None)
        train_index = self.get('train_index', None)
        test_index = self.get('test_index', None)
        if train_index is None and test_index is None:
            train_index = test_index = df.index

        # Inherit keys frow, except 'data'.
        train = Dataset(dict(self, **{'data': df.loc[train_index]}))
        test = Dataset(dict(self, **{'data': df.loc[test_index]}))

        return train, test

    # TODO: Move out or reformat.
    def dump(self, filepath, obj, **kwargs):
        """Dump predictions to disk.


        Parameters
        ----------
        filepath: str
            Target file path without extension.
        obj: object
            Object to dump.
        **kwargs

        Returns
        -------

        """
        # [deprecated]
        # raw_names = self.get('raw_names')
        if kwargs['template']:
            # recover original index and names
            obj = pd.DataFrame(index=template.index.values,
                               data={zip(template.columns, obj)}).rename_axis(template.index.name)
            # [deprecated] not enough abstract
            # df = pd.DataFrame(index=self.get('data').index.values,
            #                data={raw_names['targets'][0]: y_pred}).rename_axis(raw_names['index'])

        with open(f"{filepath}.csv", 'w', newline='') as f:
            obj.to_csv(f, mode='w', header=True,
                       index=True, sep=',', line_terminator='\n')  # only LF
        return



class DataIO(object):
    """Get raw data from database.

    Interface: get, dump_cache, load_cache

    Parameters
    ----------
    project_path: str.
        Absolute path to current project dir (with conf.py).
    logger : logger object.
        Logs.

    """
    _required_parameters = ['project_path', 'logger']

    def __init__(self, project_path, logger):
        self.logger = logger
        self.project_path = project_path

    # @time_profiler
    # @memory_profiler
    def get(self, dataset, filename='data/train.csv',
            random_skip=False, random_state=None, **kwargs):
        """Get data from csv-file.

        Parameters
        ----------
        dataset : dict
            Template for dataset.
        filename : str, optional (default='data/train.csv')
            Path to csv file reslative to `project_dir`.
        random_skip : bool, optional (default=False)
            If True randomly skip rows while read file, remains 'nrow' lines.
            Rewrite `skiprows` kwarg.
        random_state : int, None.
            Fix random state for `random_skip`.
        **kwargs : kwargs
            Additional parameter passed to the pandas.read_csv().

        Returns
        -------
        dataset : dict
            {'data': pandas.Dataframe}.

        Notes:
        ------
        If `nrow` > lines in file, auto set to None.

        """
        self.logger.info("\u25CF \u25B6 LOAD DATA")
        filename = "{}/{}".format(self.project_path, filename)
        # count lines
        with open(filename, 'r') as f:
            lines = sum(1 for _ in f)

        if 'skiprows' in kwargs and random_skip:
            self.logger.warning("random_skip rewrite skiprows kwarg.")

        nrows = kwargs.get('nrows', None)
        skiprows = kwargs.get('skipwoes', None)
        if nrows:
            if nrows > lines:
                nrows = None
            elif random_skip:
                # skiprows index strat from 0,
                # headers should be, otherwise return nrows+1.
                random_state = sklearn.utils.check_random_state(random_state)
                skiprows = random_state.choice(range(1, lines),
                                               size=lines - nrows - 1,
                                               replace=False, p=None)
            kwargs['skiprows'] = skiprows
            kwargs['nrows'] = nrows

        with open(filename, 'r') as f:
            raw = pd.read_csv(f, **kwargs)
        self.logger.info("Data loaded from:\n    {}".format(filename))

        dataset['data'] = raw
        return dataset

    def dump_cache(self, dataset, prefix,
                   fformat='pickle',
                   cachedir=None,
                   **kwargs):
        """Dump intermediate state of dataset to disk.

        Parameters
        ----------
        dataset : pickable
            Object to dump.
        prefix : str
            File identifier, added to filename.
        fformat : 'pickle'/'hr', optional default('pickle')
            If 'pickle', dump dataset via dill lib.
            If 'hr' try to decompose in csv/json (only for dictionary).
        cachedir : str, optional(default=None)
            Absolute path to dir for cache.
            If None, "project_path/results/cache/data" is used.
        **kwargs : kwargs
            Additional parameters to pass in dill.dump.\

        Returns
        -------
        dataset : pickable
            Unchanged input for compliance with producer logic.

        Notes
        -----
        'hr' means human-readable

        """
        if not cachedir:
            cachedir = f"{self.project_path}/results/cache/data"
        if not os.path.exists(cachedir):
            # create temp dir for cache if not exist
            os.makedirs(cachedir)
        for filename in glob.glob(f"{cachedir}/{prefix}*"):
            os.remove(filename)
        fps = set()
        if fformat == 'pickle':
            # pickle
            filepath = f'{cachedir}/{prefix}_.dump'
            fps.add(filepath)
            dill.dump(dataset, filepath, **kwargs)
        elif fformat == 'hr':
            filepaths = self._hr_dump(dataset, cachedir, prefix)
            fps.add(filepaths)
        else:
            raise ValueError(f"Unknown 'fformat' {fformat}.")

        self.logger.warning('Warning: update cache file(s):\n'
                            '    {}'.format('\n    '.join(fps)))
        return dataset

    def load_cache(self, dataset, prefix,
                   fformat='pickle', cachedir=None, **kwargs):
        """Load intermediate state of dataset from disk.

        Parameters
        ----------
        dataset : pickable
            Updated for 'hr', ignored for 'pickle'.
        prefix : str
            File identifier, added to filename.
        fformat : 'pickle'/'hr', optional default('pickle')
            If 'pickle', load dataset via dill lib.
            If 'hr' try to compose csv/json files in a dictionary.
        cachedir : str, optional(default=None)
            Absolute path to dir for cache.
            If None, "project_path/results/cache/data" is used.
        **kwargs : kwargs
            Additional parameters to pass in dill.dump.

        Returns
        -------
        dataset : pickable
            Loaded cache.

        """
        cachedir = f"{self.project_path}/results/cache/data"
        if fformat == 'pickle':
            filepath = f'{cachedir}/{prefix}_.dump'
            dataset = dill.load(filepath)
        elif fformat == 'hr':
            ob = self._hr_load(cachedir, prefix)
            dataset.update(ob)
        else:
            raise ValueError(f"Unknown 'fformat' {fformat}.")
        self.logger.warning(f"Warning: use cache file(s):\n    {cachedir}")
        return dataset

    def _hr_dump(self, ob, filedir, prefix):
        """Dump an dictionary to a file(s) in human-readable format.

        Traverse dictionary items and dump pandas/numpy object to separate
        csv-files, others to json-files.

        Parameters
        ----------
        ob : dict
            Object to dump.
        filedir : str
            Dump directory.
        prefix : str
            Prefix for files names.

        Returns
        -------
        filenames : set of str
            Resulted filenames "prefix_key_.ext".

        """
        if not isinstance(ob, dict):
            raise ValueError("Object should be a dictionary.")
        filenames = set()
        for key, val in ob.items():
            if isinstance(ob[key], (pd.DataFrame, pd.Series)):
                filepath = f'{filedir}/{prefix}_{key}_.csv'
                filenames.add(filepath)
                with open(filepath, 'w', newline='') as f:
                    val.to_csv(f, mode='w', header=True,
                               index=True, line_terminator='\n')
            elif isinstance(ob[key], np.ndarray):
                filepath = f'{filedir}/{prefix}_{key}_.csv'
                filenames.add(filepath)
                with open(filepath, 'w', newline='') as f:
                    pd.DataFrame(val).to_csv(f, mode='w', header=True,
                                             index=True, line_terminator='\n')
                # np.savetxt(filepath, val, delimiter=",")
            else:
                filepath = f'{filedir}/{prefix}_{key}_.json'
                filenames.add(filepath)
                with open(filepath, 'w') as f:
                    # items() preserve first level dic keys as int
                    json.dump(list(val.items()), f)
        return filenames

    def _hr_load(self, filedir, prefix):
        """Load an object from file(s) and compose in dictionary.

        Parameters
        ----------
        filedir : str
            Load directory.
        prefix : str
            Prefix for target files names.

        Returns
        -------
        ob : dict
            Resulted object.

        Notes
        -----
        Dictionary keys are gotten from filenames "prefix_key_.ext".

        """
        ob = {}
        for filepath in glob.glob(f"{filedir}/{prefix}*"):
            key = '_'.join(filepath.split('_')[1:-1])
            if filepath.endswith('.csv'):
                with open(filepath, 'r') as f:
                    ob[key] = pd.read_csv(f, sep=",", index_col=0)
            else:
                with open(filepath, 'r') as f:
                    # [alternative] object_hook=json_keys2int)
                    ob[key] = dict(json.load(f))
        return ob


class DataPreprocessor(object):
    """Transform raw data in compliance with `Dataset` class.

    Interface: preprocess, info, unify, split, check.

    Parameters
    ----------
    project_path: str.
        Absolute path to current project dir (with conf.py).
    logger : logger object.
        Logs.

    """
    _required_parameters = ['project_path', 'logger']

    # @time_profiler
    # @memory_profiler
    def __init__(self, project_path, logger):
        self.logger = logger
        self.project_path = project_path

    def preprocess(self, dataset,
                   target_names=None, categor_names=None, pos_labels=None,
                   **kwargs):
        """Preprocess raw data.

        Parameters
        ----------
        dataset : dict
            Raw dataset.
        target_names: list, None, optional (default=None)
            List of target identifiers in raw dataset.
            If None, auto add zero values under 'target' id.
            TODO: better not to auto add.
        categor_names: list, None, optional (default=None)
            List of categoric features(also binary) identifiers in raw dataset.
            If None, empty list.
        pos_labels: list, None, optional (default=None)
            Classification only, list of targets pos labels
            # TODO: what for?
            If None, empty list.
        **kwargs : kwargs
            Additional parameters to add in dataset.

        Returns
        -------
        dataset : dict
            Resulted dataset.
            Key 'data' updated.
                TODO: it is better to move unify to make_features.
            Keys added:
            'raw_names' : {}
                TODO: description after change.

        """
        self.logger.info("\u25CF \u25B6 PREPROCESS DATA")
        raw = dataset['data']
        if categor_names is None:
            categor_names = []
        if target_names is None:
            target_names = ['target']
        if pos_labels is None:
            pos_labels = []
        index = raw.index
        targets = self._make_targets(raw, target_names)
        features, features_names = self._make_features(raw, target_names)
        raw_names = {'index': list(index),
                     'index_name': index.name,
                     'targets': target_names,
                     'features': list(features_names),
                     'categor_features': categor_names,
                     'pos_labels': pos_labels}
        data = self._combine(index, targets, features, raw_names)
        dataset.update({'data': data,
                        'raw_names': raw_names,
                        **kwargs})
        return dataset

    def info(self, dataset, **kwargs):
        """Log dataset info.

        Check:
        * duplicates.
        * gaps.

        Parameters
        ----------
        dataset : dict
            Dataset to explore.
        **kwargs : kwargs
            Additional parameters to pass in low-level functions.

        Returns
        -------
        dataset : dict
            Unchanged, for compliance with producer logic.

        """
        self._check_duplicates(dataset['data'], **kwargs)
        self._check_gaps(dataset['data'], **kwargs)
        return dataset

    def unify(self, dataset, **kwargs):
        """Unify input dataset.


        Parameters
        ----------
        dataset : dict
            Dataset to unify.

        **kwargs : kwargs
            Additional parameters to pass in low-level functions.

        Notes
        -----
        python float = np.float = C double
        = np.float64 = np.double(64 bit processor)).

        Returns
        -------
        dataset: dict
            Resulted Dataset.
            Key 'data' updated:
            * fill gaps.
                if gap in categorical => fill 'unknown'
                if gap in non-categor => np.nan
            * cast categorical features to str dtype, and apply Ordinalencoder.
            * cast the whole dataframe to np.float64.

            TODO: is it possible to deprecate or move to preprocess?
                better move to raw_names and maybe reformat.
            Keys added:
            * 'categoric_ind_name' : {'column_index': ('feature_name', ['cat1', 'cat2'])}
            * 'numeric_ind_name' : {'column_index':('feature_name',)}
            dictionaries with feature indices as key, and tuple as value,
            where 'index' = column index in dataframe when drop targets.

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
                # Fill gaps with 'unknown', inplace unreliable (copy!)
                data[column_name] = data[column_name]\
                    .fillna(value='unknown', method=None, axis=None,
                            inplace=False, limit=None, downcast=None)
                # Cast dtype to str (copy!).
                data[column_name] = data[column_name].astype(str)
                # Encode
                encoder = sklearn.preprocessing.OrdinalEncoder(categories='auto')
                data[column_name] = encoder\
                    .fit_transform(data[column_name]
                                   .values.reshape(-1, 1))
                # Generate {index: ('feature_id', ['B','A','C'])}.
                # tolist need for 'hr' cache dump.
                categoric_ind_name[ind-count] = (column_name,
                                                 encoder.categories_[0].tolist())
            else:
                # Fill gaps with np.nan.
                data[column_name].fillna(value=np.nan, method=None, axis=None,
                                         inplace=True, limit=None, downcast=None)
                # Generate {'index': ('feature_id',)}.
                numeric_ind_name[ind-count] = (column_name,)
        # cast to np.float64 without copy
        # [alternative] sklearn.utils.as_float_array / assert_all_finite
        data = data.astype(np.float64, copy=False, errors='ignore')

        dataset.update({'data': data,
                        'categoric_ind_name': categoric_ind_name,
                        'numeric_ind_name': numeric_ind_name})
        # additional check
        dataset = self._check_numeric_types(dataset)
        return dataset

    # @memory_profiler
    def split(self, dataset, **kwargs):
        """Split dataset on train, test.


        Parameters
        ----------
        dataset : dict
            Dataset to unify.

        **kwargs : kwargs
            Additional parameters to pass in `sklearn.model_selection.
            train_test_split`.

        Returns
        -------
        dataset : dict
            Resulted dataset. Keys added:
            'train_index' : array-like train rows indices.
            'test_index' : array-like test rows indices.

        Notes
        -----
        If split ``train_size`` set to 1.0, test=train used.

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
            train, test, train_index, test_index = \
                sklearn.model_selection.train_test_split(
                data, data.index.values, **kwargs)

        # add to data
        dataset.update({'train_index': train_index,
                        'test_index': test_index})
        return dataset

    def _make_targets(self, raw, target_names):
        """Targets preprocessing."""
        try:
            targets_df = raw[target_names]
            targets = targets_df.values
            # TODO: test that there is no consequences.
            # [deprecated] atavism.
            # targets = targets_df.values.astype(int)  # cast to int
        except KeyError as e:
            # handle test data without targets
            self.logger.warning("Warning: no target column(s) '{}' in df,"
                                " use 0 values.".format(target_names))
            targets = np.zeros((raw.shape[0], len(target_names)),
                               dtype=int,
                               order="C")
            raw[target_names] = pd.DataFrame(targets)

        return targets

    def _make_features(self, raw, target_names):
        """Features preprocessing."""
        features_df = raw.drop(target_names, axis=1)
        raw_features_names = features_df.columns
        features = features_df.values
        return features, raw_features_names

    def _combine(self, index, targets, features, raw_names):
        """Combine preprocessed sub-data."""
        # [deprecated] preserve original
        # columns = [f'feature_categor_{i}__{raw_name}' if raw_name in raw_names['categor_features']
        #            else f'feature_{i}__{raw_name}'
        #            for i, raw_name in enumerate(raw_names['features'])]
        columns = raw_names['features']
        df = pd.DataFrame(
            data=features,
            index=index,
            columns=columns,
            copy=False,
        ).rename_axis(raw_names['index_name'])
        df.insert(loc=0, column='targets', value=targets)
        return df

    def _check_duplicates(self, data, del_duplicates=None):
        """Check duplicates rows in dataframe."""
        # Duplicate rows index mask.
        mask = data.duplicated(subset=None, keep='first')
        dupl_n = np.sum(mask)
        if dupl_n:
            self.logger.warning('Warning: {} duplicates rows found,\n'
                                '    see debug.log for details.'.format(dupl_n))
            # Count unique duplicated rows.
            rows_count = data[mask].groupby(data.columns.tolist())\
                .size().reset_index().rename(columns={0: 'count'})
            rows_count.sort_values(by=['count'], axis=0, ascending=False, inplace=True)
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                self.logger.debug('Duplicates found\n{}\n'
                                  .format(tabulate.tabulate(rows_count, headers='keys', tablefmt='psql')))

        if del_duplicates:
            # Delete duplicates.
            size_before = data.size
            data.drop_duplicates(keep='first', inplace=True)
            # TODO: if do befor preprocess is OK.
            # [deprected] not reset index, otherwise needs update raw_names)
            # data.reset_index(drop=True, inplace=True)
            size_after = data.size
            if size_before - size_after != 0:
                self.logger.warning('Warning: delete duplicates rows ({} values)'.format(size_before - size_after))

    def _check_gaps(self, data, **kwargs):
        """Check gaps in dataframe."""
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

    def _check_numeric_types(self, dataset):
        """Check that all non-categorical features are of numeric type."""
        data = dataset['data']
        raw_names = dataset['raw_names']
        dtypes = data.dtypes
        misstype = []
        for ind, column_name in enumerate(data):
            if column_name not in raw_names['categor_features']:
                if not np.issubdtype(dtypes[column_name], np.number):
                    misstype.append(column_name)
        if misstype:
            raise ValueError("Input data non-categoric columns"
                             " should be subtype of np.number, check:\n    {}".format(misstype))
        return dataset


class DataProducer(mlshell.Producer, DataIO, DataPreprocessor):
    """Class includes methods to produce dataset.

    Parameters
    ----------
    project_path: str
        Absolute path to current project dir (with conf.py).
    logger : logger object
        Logs.

    """
    _required_parameters = ['project_path', 'logger']

    def __init__(self, project_path, logger):
        self.logger = logger
        self.project_path = project_path
        mlshell.Producer.__init__(self, self.project_path, self.logger)
        DataIO.__init__(self, self.project_path, self.logger)
        DataPreprocessor.__init__(self, self.project_path, self.logger)


if __name__ == '__main__':
    pass
