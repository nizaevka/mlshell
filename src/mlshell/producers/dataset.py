""""
The :mod:`mlshell.dataset` contains examples for `Dataset` class to create
empty data object and `DataProducer` class to fulfill it.

`Dataset` class proposes unified interface to interact with underlying data.
Intended to be used in `mlshell.Workflow`. For new data formats no need to edit
`Workflow` class, only update `Dataset` interface logic. Current realization
based on dictionary.

`DataProducer` methods for convenience divided on:
* `DataIO` class to define IO related methods.
Currently implements reading from csv-file.
* `DataPreprocessor` class to preprocess data to final state.
Implements data transformation in compliance to `Dataset` and common
exploration techniques.

See also
--------
:class:`mlshell.Workflow` docstring for dataset prerequisites.

TODO: check what await for.
To use in Workflow:
get_x
get_y
get_classes (HpResolver.th_resolver)
dump
split
...

TODO:
* categoric_ind_name/numeric_ind_name used in resolver if hp_name exist and set
'auto'/['auto']



TODO: check
* categoric_ind_name/numeric_ind_name move under raw_names
* [deprecated] If None, auto add zero values under 'target' id.
        try:
            targets_df = raw[target_names]
            targets = targets_df.values
        except KeyError as e:
            # Handle test data without targets.
            self.logger.warning("Warning: no target column(s) '{}' in df,"
                                " use 0 values.".format(target_names))
            targets = np.zeros((raw.shape[0], len(target_names)),
                               dtype=int,
                               order="C")
            raw[target_names] = pd.DataFrame(targets)
* [deprecated] atavism.
targets = targets_df.values.astype(int)  # cast to int
* keys change
'index' => 'indices'
'index_name' => 'index' , also made list ['label']
'categor_features' => 'categoric_features'

"""


import copy
import os

import jsbeautifier
import numpy as np
import pandas as pd
import sklearn
import mlshell.pycnfg as pycnfg
import tabulate

__all__ = ['Dataset', 'DataIO', 'DataPreprocessor', 'DatasetProducer']


class Dataset(dict):
    """Unified data interface.

    Implements interface to access arbitrary data.
    Interface: get_x, get_y, split, dump_prediction.

    Parameters
    ----------
    *args : list
        Passed to parent class constructor.
    **kwrags : dict
        Passed to parent class constructor.

    Attributes
    ----------
    data : pd.DataFrame
        Underlying data.
    raw_names : dict
        Includes index/targets/features identifiers:
        {
            'oid': str
                Dataset identifier.
            'index': list
                List of index label(s).
            'features': list
                List of feature label(s).
            'categoric_features': list
                List of categorical feature label(s).
            'targets': list
                List of target label(s),
            'indices': list
                List of rows indices.
            'pos_labels': list, optional
                List of "positive" label(s) in target(s), classification only.
            categoric_ind_name : dict, optional
                {'column_index': ('feature_name', ['cat1', 'cat2'])}
                Dictionary with categorical feature indices as key, and tuple
                ('feature_name', categories) as value.
            numeric_ind_name : dict, optional
                {'columns_index':('feature_name',)}
                Dictionary with numeric features indices as key, and tuple
                ('feature_name', ) as value.)}
        }
    train_index : array-like, optional
        Train indices.
    test_index : array-like, optional
        Test indices.

    Notes
    -----
    Inherited from dict class, so attributes section describes keys.


    """
    _required_parameters = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __hash__(self):
        return pd.util.hash_pandas_object(self['data']).sum()

    @property
    def oid(self):
        return self['oid']

    @oid.setter
    def oid(self, value):
        self['oid'] = value

    def get_x(self):
        """Extract features from dataset.

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

        Returns
        -------
        targets : pd.DataFrame
            Extracted target columns.

        """
        df = self['data']
        raw_names = self['raw_names']
        return df[raw_names['targets']]

    def split(self):
        """Split dataset on train and test.

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
        # Inherit keys, except 'data'.
        train = Dataset(dict(self, **{'data': df.loc[train_index]}))
        test = Dataset(dict(self, **{'data': df.loc[test_index]}))
        return train, test

    def dump_prediction(self, filepath, y_pred, **kwargs):
        """Dump columns to disk.

        Parameters
        ----------
        filepath: str
            Target filepath without extension.
        y_pred: array-like
            pipeline.predict() result.
        **kwargs: dict
        `   Additional kwargs to pass in .to_csv(**kwargs).

        """
        y_true = self.get_y()
        # Recover original index and names.
        obj = pd.DataFrame(index=y_true.index.values,
                           data={zip(y_true.columns, y_pred)})\
            .rename_axis(y_true.index.name)
        with open(f"{filepath}.csv", 'w', newline='') as f:
            obj.to_csv(f, mode='w', header=True,
                       index=True, sep=',', line_terminator='\n', **kwargs)
        return None


class DataIO(object):
    """Get raw data from database.

    Interface: get.

    Parameters
    ----------
    project_path: str.
        Absolute path to current project dir.
    logger : logger object.
        Logs.

    """
    _required_parameters = ['project_path', 'logger']

    def __init__(self, project_path, logger):
        self.logger = logger
        self.project_path = project_path

    # @time_profiler
    # @memory_profiler
    def load(self, dataset, filepath,
             random_skip=False, random_state=None, **kwargs):
        """Load data from csv-file.

        Parameters
        ----------
        dataset : Dataset
            Template for dataset.
        filepath : str, optional
            Absolute path to csv file or relative to 'self.project_dir' started
            with './'.
        random_skip : bool, optional (default=False)
            If True randomly skip rows while read file, remains 'nrow' lines.
            Rewrite `skiprows` kwarg.
        random_state : int, None.
            Fix random state for `random_skip`.
        **kwargs : kwargs
            Additional parameter passed to the pandas.read_csv().

        Returns
        -------
        dataset : Dataset
            Key added {'data': pandas.DataFrame}.

        Notes:
        ------
        If `nrow` > lines in file, auto set to None.

        """
        self.logger.info("    |__ LOAD DATA")
        if filepath.startswith('./'):
            filepath = "{}/{}".format(self.project_path, filepath[2:])

        # Count lines.
        with open(filepath, 'r') as f:
            lines = sum(1 for _ in f)
        if 'skiprows' in kwargs and random_skip:
            self.logger.warning("random_skip rewrite skiprows kwarg.")
        nrows = kwargs.get('nrows', None)
        skiprows = kwargs.get('skipwoes', None)
        if nrows:
            if nrows > lines:
                nrows = None
            elif random_skip:
                # skiprows index start from 0.
                # If no headers, returns nrows+1.
                random_state = sklearn.utils.check_random_state(random_state)
                skiprows = random_state.choice(range(1, lines),
                                               size=lines - nrows - 1,
                                               replace=False, p=None)
            kwargs['skiprows'] = skiprows
            kwargs['nrows'] = nrows
        with open(filepath, 'r') as f:
            raw = pd.read_csv(f, **kwargs)
        self.logger.info("Data loaded from:\n    {}".format(filepath))
        dataset['data'] = raw
        return dataset


class DataPreprocessor(object):
    """Transform raw data in compliance with `Dataset` class.

    Interface: preprocess, info, split.

    Parameters
    ----------
    project_path: str.
        Absolute path to current project dir.
    logger : logger object.
        Logs.

    """
    _required_parameters = ['project_path', 'logger']

    # @time_profiler
    # @memory_profiler
    def __init__(self, project_path, logger):
        self.logger = logger
        self.project_path = project_path

    def preprocess(self, dataset, targets_names, features_names=None,
                   categor_names=None,
                   pos_labels=None, **kwargs):
        """Preprocess raw data.

        Parameters
        ----------
        dataset : Dataset {'data': pandas.DataFrame}
            Raw dataset.
        targets_names: list
            List of target identifiers in raw dataset.
        features_names TODO
        categor_names: list, None, optional (default=None)
            List of categoric features(also binary) identifiers in raw dataset.
            If None, empty list.
        pos_labels: list, None, optional (default=None)
            Classification only, list of "positive" labels for targets.
            Could be used for threshold analysis (roc_curve) and metrics
            evaluation if classifiers supported predict_proba. If None, last
            label in np.unique(target) for each target is used.
        **kwargs : kwargs
            Additional parameters to add in dataset.

        Returns
        -------
        dataset : Dataset
            Resulted dataset. Key 'data' updated. Key added:
            'raw_names' : {
                'index': list
                    List of index label(s).
                'features': list
                    List of feature label(s).
                'categoric_features': list
                    List of categorical feature label(s).
                'targets': list
                    List of target label(s),
                'indices': list
                    List of rows indices.
                'classes': list
                    List of unique labels for each target.
                'pos_labels': list
                    List of "positive" label(s) for each target.
                'pos_labels_ind': list
                    List of "positive" label(s) index in np.unique(target) for
                    each target.
                categoric_ind_name : dict
                    {'column_index': ('feature_name', ['cat1', 'cat2'])}
                    Dictionary with categorical feature indices as key, and
                    tuple ('feature_name', categories) as value.
                numeric_ind_name : dict {'columns_index':('feature_name',)}
                    Dictionary with numeric features indices as key, and tuple
                    ('feature_name', ) as value.)}
            }

        Notes
        -----
        Don`t change dataframe shape or index/columns names after `raw_names`
        generating.

        """
        self.logger.info("    |__ PREPROCESS DATA")
        raw = dataset['data']
        if categor_names is None:
            categor_names = []
        if features_names is None:
            features_names = [c for c in raw.columns if c not in targets_names]
        if pos_labels is None:
            pos_labels = []

        index = raw.index
        targets, raw_info_targets =\
            self._process_targets(raw, targets_names, pos_labels)
        features, raw_info_features =\
            self._process_features(raw, features_names, categor_names)
        raw_names = {
            'index': index.name,
            'indices': list(index),
            'targets': targets_names,
            'features': list(features_names),
            'categoric_features': categor_names,
            **raw_info_features,
            **raw_info_targets,
        }
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
        dataset : Dataset
            Dataset to explore.
        **kwargs : kwargs
            Additional parameters to pass in low-level functions.

        Returns
        -------
        dataset : Dataset
            For compliance with producer logic.

        """
        self._check_duplicates(dataset['data'], **kwargs)
        self._check_gaps(dataset['data'], **kwargs)
        return dataset

    # @memory_profiler
    def split(self, dataset, **kwargs):
        """Split dataset on train, test.


        Parameters
        ----------
        dataset : Dataset
            Dataset to unify.

        **kwargs : kwargs
            Additional parameters to pass in `sklearn.model_selection.
            train_test_split`.

        Returns
        -------
        dataset : Dataset
            Resulted dataset. Keys added:
            {'train_index' : array-like train rows indices,
             'test_index' : array-like test rows indices.}

        Notes
        -----
        If split ``train_size`` set to 1.0, test=train used.

        """
        self.logger.info("|__  SPLIT DATA")
        data = dataset['data']

        if (kwargs['train_size'] == 1.0 and kwargs['test_size'] is None
                or kwargs['train_size'] is None and kwargs['test_size'] == 0):
            # train = test = data
            train_index = test_index = data.index
        else:
            shell_kw = ['func']
            kwargs = copy.deepcopy(kwargs)
            for kw in shell_kw:
                kwargs.pop(kw)
            train, test, train_index, test_index = \
                sklearn.model_selection.train_test_split(
                    data, data.index.values, **kwargs)

        # Add to dataset.
        dataset.update({'train_index': train_index,
                        'test_index': test_index})
        return dataset

    def _process_targets(self, raw, target_names, pos_labels):
        """Targets preprocessing."""
        targets_df = raw[target_names]
        targets_df, classes, pos_labels, pos_labels_ind =\
            self._unify_targets(targets_df, pos_labels)
        targets = targets_df.values
        raw_info_targets = {
            'classes': classes,
            'pos_labels': pos_labels,
            'pos_labels_ind': pos_labels_ind,
        }
        return targets, raw_info_targets

    def _process_features(self, raw, features_names, categor_names):
        """Features preprocessing."""
        features_df = raw[features_names]
        features_df, categoric_ind_name, numeric_ind_name \
            = self._unify_features(features_df, categor_names)
        features = features_df.values
        raw_info_features = {
            'categoric_ind_name': categoric_ind_name,
            'numeric_ind_name': numeric_ind_name,}
        return features, raw_info_features

    def _combine(self, index, targets, features, raw_names):
        """Combine preprocessed sub-data."""
        columns = raw_names['features']
        df = pd.DataFrame(
            data=features,
            index=index,
            columns=columns,
            copy=False,
        ).rename_axis(raw_names['index'])
        df.insert(loc=0, column='targets', value=targets)
        return df

    def _unify_targets(self, targets, pos_labels=None):
        """Unify input targets.

        Extract classes and positive label index (classification only).

        Parameters
        ----------
        targets : pd.DataFrame
            Data to unify.
        pos_labels: list, None, optional (default=None)
            Classification only, list of "positive" labels for targets.
            Could be used for threshold analysis (roc_curve) and metrics
            evaluation if classifiers supported predict_proba. If None, last
            label in np.unique(target) for each target is used.

        Returns
        -------
        targets: pd.DataFrame
            Unchanged input.
        'classes': list
            List of labels for each target.
        'pos_labels': list
            List of "positive" label(s) for each target.
        'pos_labels_ind': list
            List of "positive" label(s) index in np.unique(target) for
            each target.

        """
        # Find classes, example: [array([1]), array([2, 7])].
        classes = [np.unique(j) for i, j in targets.iteritems()]
        if not pos_labels:
            pos_labels_ind = -1
            pos_labels = [i[-1] for i in classes]  # [2,4]
        else:
            # Find where pos_labels in sorted labels, example: [1, 0].
            pos_labels_ind = [np.where(classes[i] == pos_labels[i])[0][0]
                              for i in range(len(classes))]
        self.logger.info(
            f"Labels {pos_labels} identified as positive:\n"
            f"    for classifiers supported predict_proba:"
            f" if P(pos_labels)>threshold, prediction=pos_labels on sample.")
        return targets, classes, pos_labels, pos_labels_ind

    def _unify_features(self, features, categor_names):
        """Unify input features.

        Parameters
        ----------
        features : pd.DataFrame
            Data to unify.
        categor_names: list
            List of categorical features (and binary) column names in features.

        Returns
        -------
        features: pd.DataFrame
            Input updates:
            * fill gaps.
                if gap in categorical => fill 'unknown'
                if gap in non-categor => np.nan
            * cast categorical features to str dtype, and apply Ordinalencoder.
            * cast the whole featuresframe to np.float64.
        categoric_ind_name : dict
            {'column_index': ('feature_name', ['cat1', 'cat2'])}
            Dictionary with categorical feature indices as key, and tuple
            ('feature_name', categories) as value.
        numeric_ind_name : dict {'columns_index':('feature_name',)}
            Dictionary with numeric features indices as key, and tuple
            ('feature_name', ) as value.

        """
        categoric_ind_name = {}
        numeric_ind_name = {}
        for ind, column_name in enumerate(features):
            if column_name in categor_names:
                # Fill gaps with 'unknown', inplace unreliable (copy!)
                features[column_name] = features[column_name]\
                    .fillna(value='unknown', method=None, axis=None,
                            inplace=False, limit=None, downcast=None)
                # Cast dtype to str (copy!).
                features[column_name] = features[column_name].astype(str)
                # Encode
                encoder = sklearn.preprocessing.\
                    OrdinalEncoder(categories='auto')
                features[column_name] = encoder\
                    .fit_transform(features[column_name]
                                   .values.reshape(-1, 1))
                # Generate {index: ('feature_id', ['B','A','C'])}.
                # tolist need for 'hr' cache dump.
                categoric_ind_name[ind] = (column_name,
                                           encoder.categories_[0].tolist())
            else:
                # Fill gaps with np.nan.
                features[column_name].fillna(value=np.nan, method=None, axis=None,
                                         inplace=True, downcast=None)
                # Generate {'index': ('feature_id',)}.
                numeric_ind_name[ind] = (column_name,)
        # Cast to np.float64 without copy.
        # python float = np.float = C double =
        # np.float64 = np.double(64 bit processor)).
        # [alternative] sklearn.utils.as_float_array / assert_all_finite
        features = features.astype(np.float64, copy=False, errors='ignore')
        # Additional check.
        self._check_numeric_types(features, categor_names)
        return features, categoric_ind_name, numeric_ind_name

    def _check_duplicates(self, data, del_duplicates=False):
        """Check duplicates rows in dataframe.

        Parameters
        ----------
        data : pandas.DataFrame
            Dataframe to check.
        del_duplicates : bool
            If True, delete rows with duplicated.
            If False, do nothing.

        Notes
        -----
        Use del_duplicates=True only before generating dataset `raw_names`.

        """
        # Duplicate rows index mask.
        mask = data.duplicated(subset=None, keep='first')
        dupl_n = np.sum(mask)
        if dupl_n:
            self.logger.warning(f"Warning: {dupl_n} duplicates rows found,\n"
                                "    see debug.log for details.")
            # Count unique duplicated rows.
            rows_count = data[mask].groupby(data.columns.tolist())\
                .size().reset_index().rename(columns={0: 'count'})
            rows_count.sort_values(by=['count'], axis=0,
                                   ascending=False, inplace=True)
            with pd.option_context('display.max_rows', None,
                                   'display.max_columns', None):
                pprint = tabulate.tabulate(rows_count, headers='keys',
                                           tablefmt='psql')
                self.logger.debug(f"Duplicates found\n{pprint}")

        if del_duplicates:
            # Delete duplicates (without index reset).
            size_before = data.size
            data.drop_duplicates(keep='first', inplace=True)
            size_after = data.size
            if size_before - size_after != 0:
                self.logger.warning(f"Warning: delete duplicates rows "
                                    f"({size_before - size_after} values).")
        return None

    def _check_gaps(self, data, del_gaps=False, nogap_columns=None):
        """Check gaps in dataframe.

        Parameters
        ----------
        data : pandas.DataFrame
            Dataframe to check.
        del_gaps : bool
            If True, delete rows with gaps from `nongap_columns` list.
            If False, raise Exception when `nongap_columns` contain gaps.
        nogap_columns : list ['column_1', ..]
            Columns where gaps are forbidden. if None, empty.

        Notes
        -----
        Use del_geps=True only before generating dataset `raw_names`.

        """
        gaps_number = data.size - data.count().sum()
        columns_with_gaps_dic = {}
        if gaps_number > 0:
            for column_name in data:
                column_gaps_namber = data[column_name].size \
                                     - data[column_name].count()
                if column_gaps_namber > 0:
                    columns_with_gaps_dic[column_name] = column_gaps_namber
            self.logger.warning('Warning: gaps found: {} {:.3f}%,\n'
                                '    see debug.log for details.'
                                .format(gaps_number, gaps_number / data.size))
            pprint = jsbeautifier.beautify(str(columns_with_gaps_dic))
            self.logger.debug(f"Gaps per column:\n{pprint}")

        subset = [column_name for column_name in nogap_columns
                  if column_name in columns_with_gaps_dic]
        if del_gaps and subset:
            # Delete rows with gaps in specified columns.
            data.dropna(axis=0, how='any', thresh=None,
                        subset=[subset], inplace=True)
        elif subset:
            raise ValueError(f"Gaps in {subset}.")
        return None

    def _check_numeric_types(self, data, categor_names):
        """Check that all non-categorical features are of numeric type."""
        dtypes = data.dtypes
        misstype = []
        for ind, column_name in enumerate(data):
            if column_name not in categor_names:
                if not np.issubdtype(dtypes[column_name], np.number):
                    misstype.append(column_name)
        if misstype:
            raise ValueError(f"Input data non-categoric columns should be "
                             f"subtype of np.number, check:\n"
                             f"    {misstype}")
        return None


class DatasetProducer(pycnfg.Producer, DataIO, DataPreprocessor):
    """Class includes methods to produce dataset.

    Parameters
    ----------
    objects : dict {'section_id__config__id', object,}
        Dictionary with resulted objects from previous executed producers.
    oid : str
        Unique identifier of produced object.
    path_id : str
        Project path identifier in `objects`.
    logger_id : str
        Logger identifier in `objects`.

    Attributes
    ----------
    objects : dict {'section_id__config__id', object,}
        Dictionary with resulted objects from previous executed producers.
    oid : str
        Unique identifier of produced object.
    logger : logger object
        Default logger logging.getLogger().
    project_path: str
        Absolute path to project dir.

    """
    _required_parameters = ['objects', 'oid', 'path_id', 'logger_id']

    def __init__(self, objects, oid, path_id, logger_id):
        pycnfg.Producer.__init__(self, objects, oid)
        self.logger = objects[logger_id]
        self.project_path = objects[path_id]
        DataIO.__init__(self, self.project_path, self.logger)
        DataPreprocessor.__init__(self, self.project_path, self.logger)


if __name__ == '__main__':
    pass
