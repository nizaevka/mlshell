"""
The :mod:`mlshell.producers.dataset` contains examples of `Dataset` class for
empty data object creation and `DataProducer` class for filling it.

:class:`mlshell.Dataset` proposes unified interface to interact with underlying
data. Intended to be used in :class:`mlshell.Workflow`. For new data formats
no need to edit `Workflow` class, adapt `Dataset` in compliance to interface.
Current realization based on dictionary.

:class:`mlshell.DataProducer` specifies methods divided for convenience on:

* :class:`mlshell.DataIO` defining IO related methods.
Currently reading from csv-file implemented.

* :class:`mlshell.DataPreprocessor` preprocessing data to final state.
Implemented data transformation in compliance to `Dataset` class, also common
exploration techniques available.

"""


import copy
import os

import jsbeautifier
import numpy as np
import pandas as pd
import pycnfg
import sklearn
import tabulate

__all__ = ['Dataset', 'DataIO', 'DataPreprocessor', 'DatasetProducer']


class Dataset(dict):
    """Unified data interface.

    Implements interface to access arbitrary data.

    Interface: x, y, data, meta, subset, dump_pred and whole dict api.

    Parameters
    ----------
    *args : list
        Passed to parent class constructor.
    **kwrags : dict
        Passed to parent class constructor.

    Attributes
    ----------
    data : :class:`pandas.DataFrame`
        Underlying data.
    subsets : dict
        {'subset_id' : array-like subset indices, ..}.
    meta : dict
        Extracted auxiliary information from data: {

        'index': list
            List of index column label(s).
        'features': list
            List of feature column label(s).
        'categoric_features': list
            List of categorical feature column label(s).
        'targets': list
            List of target column label(s),
        'indices': list
            List of rows indices.
        'classes': list of :class:`numpy.ndarray`
            List of sorted unique labels for each target(s) (n_outputs,
            n_classes).
        'pos_labels': list
            List of "positive" label(s) for target(s) (n_outputs,).
        'pos_labels_ind': list
            List of "positive" label(s) index in :func:`numpy.unique`
            for target(s) (n_outputs).
        categoric_ind_name : dict
            Dictionary with categorical feature indices as key, and
            tuple ('feature_name', categories) as value:
            {'column_index': ('feature_name', ['cat1', 'cat2'])}.
        numeric_ind_name : dict
            Dictionary with numeric features indices as key, and tuple
            ('feature_name', ) as value: {'columns_index':('feature_name',)}.

        }

    Notes
    -----
    Inherited from dict class, so attributes section describes keys.

    """
    _required_parameters = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __hash__(self):
        return hash(pd.util.hash_pandas_object(self['data']).sum())

    @property
    def oid(self):
        """str: Dataset identifier."""
        return self['_oid']

    @oid.setter
    def oid(self, value):
        self['_oid'] = value

    @property
    def x(self):
        """:class:`pandas.DataFrame` : Extracted features columns."""
        df = self['data']
        meta = self['meta']
        return df.loc[:, meta['features']]

    @property
    def y(self):
        """:class:`pandas.DataFrame` : Extracted targets columns."""
        df = self['data']
        meta = self['meta']
        # return df[meta['targets']].values
        res = df.loc[:, meta['targets']].values.ravel() \
            if len(meta['targets']) == 1 else df.loc[:, meta['targets']].values
        return res

    @property
    def meta(self):
        """dict: Access meta."""
        return self['meta']

    @property
    def data(self):
        """:class:`pandas.DataFrame` : Access data."""
        return self['data']

    def subset(self, subset_id):
        """:class:`mlshell.Dataset` : Access subset. """
        if subset_id is '':
            return self

        df = self['data']
        index = self['subsets'][subset_id]  # subset of meta['inices']
        # Inherit only meta (except indices).
        # dict(self) will inherit by ref.
        dataset = Dataset(**{
            'meta': copy.deepcopy(self.meta),
            'data': df.loc[index],
            'subsets': {},
            '_oid': f"{self['_oid']}__{subset_id}"})
        # Update indices in meta.
        dataset.meta['indices'] = index
        # if reset_index: np.array(dataset.meta['indices'])[index].tolist()
        return dataset

    def dump_pred(self, filepath, y_pred, **kwargs):
        """Dump columns to disk.

        Parameters
        ----------
        filepath: str
            File path without extension.
        y_pred: array-like
            pipeline.predict() result.
        **kwargs: dict
        `   Additional kwargs to pass in .to_csv(**kwargs).

        Returns
        -------
        fullpath : str
            Full filepath.

        """
        meta = self.meta
        # Recover original index and names.
        dic = dict(zip(
            meta['targets'],
            [y_pred] if len(meta['targets']) == 1 else np.array(y_pred).T
        ))
        obj = pd.DataFrame(index=meta['indices'],
                           data=dic).rename_axis(meta['index'], axis=0)
        fullpath = f"{filepath}_pred.csv"
        if "PYTEST_CURRENT_TEST" in os.environ:
            if 'float_format' not in kwargs:
                kwargs['float_format'] = '%.8f'
        with open(fullpath, 'w', newline='') as f:
            obj.to_csv(f, mode='w', header=True, index=True, sep=',',
                       line_terminator='\n', **kwargs)
        return fullpath


class DataIO(object):
    """Get raw data from database.

    Interface: load.

    Parameters
    ----------
    project_path: str.
        Absolute path to current project dir.
    logger : :class:`logging.Logger`
        Logger.

    """
    _required_parameters = ['project_path', 'logger']

    def __init__(self, project_path, logger):
        self.logger = logger
        self.project_path = project_path

    def load(self, dataset, filepath, key='data',
             random_skip=False, random_state=None, **kwargs):
        """Load data from csv-file.

        Parameters
        ----------
        dataset : :class:`mlshell.Dataset`
            Template for dataset.
        filepath : str
            Absolute path to csv file or relative to 'project__path' started
            with './'.
        key : str, optional (default='data')
            Loaded data identifier to add in dataset dictionary. Useful when
            load multiple files and combine them in separate step under 'data'.
        random_skip : bool, optional (default=False)
            If True randomly skip rows while read file, remain 'nrow' lines.
            Rewrite `skiprows` kwarg.
        random_state : int, optional (default=None).
            Fix random state for `random_skip`.
        **kwargs : dict
            Additional parameter passed to the :func:`pandas.read_csv()` .

        Returns
        -------
        dataset : :class:`mlshell.Dataset`
            Key added: {'data': :class:`pandas.DataFrame` ,}.

        Notes:
        ------
        If `nrow` > lines in file, auto set to None.

        """
        if filepath.startswith('./'):
            filepath = "{}/{}".format(self.project_path, filepath[2:])

        # Count lines.
        with open(filepath, 'r') as f:
            lines = sum(1 for _ in f)
        if 'skiprows' in kwargs and random_skip:
            self.logger.warning("random_skip rewrite skiprows kwarg.")
        nrows = kwargs.get('nrows', None)
        skiprows = kwargs.get('skiprows', None)
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
        dataset[key] = raw
        return dataset


class DataPreprocessor(object):
    """Transform raw data in compliance with `Dataset` class.

    Interface: preprocess, info, split.

    Parameters
    ----------
    project_path: str.
        Absolute path to current project dir.
    logger : :class:`logging.Logger`
        Logger.

    """
    _required_parameters = ['project_path', 'logger']

    def __init__(self, project_path, logger):
        self.logger = logger
        self.project_path = project_path

    def preprocess(self, dataset, targets_names, features_names=None,
                   categor_names=None, pos_labels=None, **kwargs):
        """Preprocess raw data.

        Parameters
        ----------
        dataset : :class:`mlshell.Dataset`
            Raw dataset: {'data': :class:`pandas.DataFrame` }.
        targets_names: list
            List of targets columns names in raw dataset. Even if no exist,
            will be used to name predictions in ``dataset.dump_pred``  .
        features_names: list, optional (default=None)
            List of features columns names in raw dataset. If None, all except
            targets.
        categor_names: list, optional (default=None)
            List of categorical features(also binary) identifiers in raw
            dataset. If None, empty list.
        pos_labels: list, optional (default=None)
            Classification only, list of "positive" label(s) in target(s).
            Could be used in :func:`sklearn.metrics.roc_curve` for
            threshold analysis and metrics evaluation if classifier supports
            ``predict_proba``. If None, for each target last label in
            :func:`numpy.unique` is used . For regression set [] to prevent
            evaluation.
        **kwargs : dict
            Additional parameters to add in dataset.

        Returns
        -------
        dataset : :class:`mlshell.Dataset`
            Resulted dataset. Key updated: 'data'. Keys added:

            'subsets': dict
                Storage for data subset(s) indices (filled in split method)
                {'subset_id': indices}.
            'meta' : dict
                Extracted auxiliary information from data:
                {

                'index': list
                    List of index column label(s).
                'features': list
                    List of feature column label(s).
                'categoric_features': list
                    List of categorical feature column label(s).
                'targets': list
                    List of target column label(s),
                'indices': list
                    List of rows indices.
                'classes': list of :class:`numpy.ndarray`
                    List of sorted unique labels for each target(s) (n_outputs,
                    n_classes).
                'pos_labels': list
                    List of "positive" label(s) for target(s) (n_outputs,).
                'pos_labels_ind': list
                    List of "positive" label(s) index in :func:`numpy.unique`
                    for target(s) (n_outputs).
                categoric_ind_name : dict
                    Dictionary with categorical feature indices as key, and
                    tuple ('feature_name', categories) as value:
                    {'column_index': ('feature_name', ['cat1', 'cat2'])}.
                numeric_ind_name : dict
                    Dictionary with numeric features indices as key, and tuple
                    ('feature_name', ) as value: {'columns_index':
                    ('feature_name',)}.

                }

        Notes
        -----
        Don`t change dataframe shape or index/columns names after ``meta``
        generating.

        Features columns unified:

        * Fill gaps.

            * If gap in categorical => set 'unknown'.
            * If gap in non-categorical => set np.nan.

        * Cast categorical features to str dtype, and apply Ordinal encoder.
        * Cast values to np.float64.

        """
        raw = dataset['data']
        if categor_names is None:
            categor_names = []
        if features_names is None:
            features_names = [c for c in raw.columns if c not in targets_names]
        for i in (targets_names, features_names, categor_names):
            if not isinstance(i, list):
                raise TypeError(f"{i} should be a list.")

        index = raw.index
        targets_df, raw_info_targets =\
            self._process_targets(raw, targets_names, pos_labels)
        features_df, raw_info_features =\
            self._process_features(raw, features_names, categor_names)
        data = self._combine(index, targets_df, features_df)
        meta = {
            'index': index.name,
            'indices': list(index),
            'targets': targets_names,
            'features': list(features_names),
            'categoric_features': categor_names,
            **raw_info_features,
            **raw_info_targets,
        }
        self.logger.debug(f"Dataset meta:\n    {meta}")
        dataset.update({'data': data,
                        'meta': meta,
                        'subsets': {},
                        **kwargs})
        return dataset

    def info(self, dataset, **kwargs):
        """Log dataset info.

        Check:

        * duplicates.
        * gaps.

        Parameters
        ----------
        dataset : :class:`mlshell.Dataset`
            Dataset to explore.
        **kwargs : dict
            Additional parameters to pass in low-level functions.

        Returns
        -------
        dataset : :class:`mlshell.Dataset`
            For compliance with producer logic.

        """
        self._check_duplicates(dataset['data'], **kwargs)
        self._check_gaps(dataset['data'], **kwargs)
        return dataset

    def split(self, dataset, **kwargs):
        """Split dataset on train, test.

        Parameters
        ----------
        dataset : :class:`mlshell.Dataset`
            Dataset to unify.

        **kwargs : dict
            Additional parameters to pass in:
            :func:`sklearn.model_selection.train_test_split` .

        Returns
        -------
        dataset : :class:`mlshell.Dataset`
            Resulted dataset. 'subset' value updated:
            {'train': array-like train rows indices,
            'test': array-like test rows indices,}

        Notes
        -----
        If split ``train_size==1.0``  or ``test_size==0``: ``test=train`` ,
        other kwargs ignored.

        No copy takes place.

        """
        if 'test_size' not in kwargs:
            kwargs['test_size'] = None
        if 'train_size' not in kwargs:
            kwargs['train_size'] = None
        data = dataset['data']

        if (kwargs['train_size'] == 1.0 and kwargs['test_size'] is None
                or kwargs['train_size'] is None and kwargs['test_size'] == 0):
            # train = test = data
            train_index = test_index = data.index
        else:
            train, test, train_index, test_index = \
                sklearn.model_selection.train_test_split(
                    data, data.index.values, **kwargs)

        # Add to dataset.
        dataset['subsets'].update({'train': train_index,
                                   'test': test_index})
        return dataset

    # ============================== preprocess ===============================
    def _process_targets(self, raw, target_names, pos_labels):
        """Targets preprocessing."""
        try:
            targets_df = raw[target_names]
        except KeyError:
            self.logger.warning(f"No target column(s) found in df:\n"
                                f"    {target_names}")
            targets_df = pd.DataFrame()
        targets_df, classes, pos_labels, pos_labels_ind =\
            self._unify_targets(targets_df, pos_labels)
        # targets = targets_df.values
        raw_info_targets = {
            'classes': classes,
            'pos_labels': pos_labels,
            'pos_labels_ind': pos_labels_ind,
        }
        return targets_df, raw_info_targets

    def _process_features(self, raw, features_names, categor_names):
        """Features preprocessing."""
        features_df = raw[features_names]
        features_df, categoric_ind_name, numeric_ind_name \
            = self._unify_features(features_df, categor_names)
        # features = features_df.values
        raw_info_features = {
            'categoric_ind_name': categoric_ind_name,
            'numeric_ind_name': numeric_ind_name, }
        return features_df, raw_info_features

    def _combine(self, index, targets_df, features_df):
        """Combine preprocessed sub-data."""
        # targets_df empty dataframe or None is possible
        return pd.concat(
            [targets_df, features_df],
            axis=1,
        )

    def _unify_targets(self, targets, pos_labels=None):
        """Unify input targets.

        Extract classes and positive label index (classification only).

        Parameters
        ----------
        targets : :class:`pandas.DataFrame`
            Data to unify.
        pos_labels: list, optional (default=None)
            Classification only, list of "positive" labels for targets.
            Could be used for threshold analysis (roc_curve) and metrics
            evaluation if classifiers supported predict_proba. If None, last
            label in :func:`numpy.unique` for each target used. For regression
            set [] to prevent evaluation.

        Returns
        -------
        targets: :class:`pandas.DataFrame`
            Unchanged input.
        classes: list of :class:`numpy.ndarray`
            List of sorted unique labels for target(s) (n_outputs, n_classes).
        pos_labels: list
            List of "positive" label(s) for target(s) (n_outputs,).
        pos_labels_ind: list
            List of "positive" label(s) index in :func:`numpy.unique`
            for target(s) (n_outputs,).

        """
        # Regression.
        if isinstance(pos_labels, list) and not pos_labels:
            classes = []
            pos_labels_ind = []
            return targets, classes, pos_labels, pos_labels_ind

        # Classification.
        # Find classes, example: [array([1]), array([2, 7])].
        classes = [np.unique(j) for i, j in targets.iteritems()]

        if pos_labels is None:
            n_targets = len(classes)
            pos_labels_ind = [len(classes[i]) - 1 for i in range(n_targets)]
            pos_labels = [classes[i][pos_labels_ind[i]]
                          for i in range(n_targets)]  # [2,4]
        else:
            # Find where pos_labels in sorted labels, example: [1, 0].
            pos_labels_ind = [np.where(classes[i] == pos_labels[i])[0][0]
                              for i in range(len(classes))]
        # Could be no target columns in new data.
        self.logger.debug(
            f"Labels {pos_labels} identified as positive for target(s):\n"
            f"    when classifier supports predict_proba: prediction="
            f"pos_label on sample, if P(pos_label) > classification "
            f"threshold.")
        return targets, classes, pos_labels, pos_labels_ind

    def _unify_features(self, features, categor_names):
        """Unify input features.

        Parameters
        ----------
        features : :class:`pandas.DataFrame`
            Data to unify.
        categor_names: list
            List of categorical features (and binary) column names in features.

        Returns
        -------
        features: :class:`pandas.DataFrame`
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
        # Turn off: SettingWithCopy, excessive.
        pd.options.mode.chained_assignment = None
        for ind, column_name in enumerate(features):
            if column_name in categor_names:
                # Fill gaps with 'unknown', inplace unreliable (copy!).
                features.loc[:, column_name] = features[column_name]\
                    .fillna(value='unknown', method=None, axis=None,
                            inplace=False, limit=None, downcast=None)
                # Cast dtype to str (copy!).
                features.loc[:, column_name] = features[column_name].astype(str)
                # Encode
                encoder = sklearn.preprocessing.\
                    OrdinalEncoder(categories='auto')
                features.loc[:, column_name] = encoder\
                    .fit_transform(features[column_name]
                                   .values.reshape(-1, 1))
                # Generate {index: ('feature_id', ['B','A','C'])}.
                # tolist need for 'hr' cache dump.
                categoric_ind_name[ind] = (column_name,
                                           encoder.categories_[0].tolist())
            else:
                # Fill gaps with np.nan, inplace unreliable (copy!).
                # Could work with no copy on slice or single col even inplace.
                features.loc[:, column_name] = features.loc[:, column_name]\
                    .fillna(value=np.nan, method=None, axis=None,
                            inplace=False, downcast=None)
                # Generate {'index': ('feature_id',)}.
                numeric_ind_name[ind] = (column_name,)
        # Turn on: SettingWithCopy.
        pd.options.mode.chained_assignment = 'warn'
        # Cast to np.float64 without copy.
        # python float = np.float = C double =
        # np.float64 = np.double(64 bit processor)).
        # [alternative] sklearn.utils.as_float_array / assert_all_finite
        features = features.astype(np.float64, copy=False, errors='ignore')
        # Additional check.
        self._check_numeric_types(features, categor_names)
        return features, categoric_ind_name, numeric_ind_name

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

    # ================================ info ===================================
    def _check_duplicates(self, data, del_duplicates=False):
        """Check duplicates rows in dataframe.

        Parameters
        ----------
        data : :class:`pandas.DataFrame`
            Dataframe to check.
        del_duplicates : bool
            If True, delete rows with duplicated.
            If False, do nothing.

        Notes
        -----
        Use del_duplicates=True only before generating dataset `meta`.

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
        data : :class:`pandas.DataFrame`
            Dataframe to check.
        del_gaps : bool, optional (default=False)
            If True, delete rows with gaps from `nongap_columns` list.
            If False, raise Exception when `nongap_columns` contain gaps.
        nogap_columns : list, optional (default=None)
            Columns where gaps are forbidden: ['column_1', ..]. if None, [].

        Notes
        -----
        Use del_geps=True only before generating dataset `meta` (preprocess).

        """
        if nogap_columns is None:
            nogap_columns = []

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


class DatasetProducer(pycnfg.Producer, DataIO, DataPreprocessor):
    """Factory to produce dataset.

    Parameters
    ----------
    objects : dict
        Dictionary with objects from previous executed producers:
        {'section_id__config__id', object,}.
    oid : str
        Unique identifier of produced object.
    path_id : str, optional (default='default')
        Project path identifier in `objects`.
    logger_id : str, optional (default='default')
        Logger identifier in `objects`.

    Attributes
    ----------
    objects : dict
        Dictionary with objects from previous executed producers:
        {'section_id__config__id', object,}.
    oid : str
        Unique identifier of produced object.
    logger : :class:`logging.Logger`
        Logger.
    project_path: str
        Absolute path to project dir.

    """
    _required_parameters = ['objects', 'oid', 'path_id', 'logger_id']

    def __init__(self, objects, oid, path_id='path__default',
                 logger_id='logger__default'):
        pycnfg.Producer.__init__(self, objects, oid, path_id=path_id,
                                 logger_id=logger_id)
        DataIO.__init__(self, self.project_path, self.logger)
        DataPreprocessor.__init__(self, self.project_path, self.logger)


if __name__ == '__main__':
    pass
