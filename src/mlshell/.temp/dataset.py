def _combine(self, index, targets, features, meta):
    """Combine preprocessed sub-data."""
    # [deprecated] not need to recreate.
    return pd.concat(
        [
            pd.DataFrame(
                data=targets,
                index=index,
                columns=meta['targets'],
                copy=False,
            ),
            pd.DataFrame(
                data=features,
                index=index,
                columns=meta['features'],
                copy=False,
            ).rename_axis(meta['index']),
        ],
        axis=1,
    )
    # [deprecated] only for one column.
    # df.insert(loc=0, column=meta['targets'], value=targets)

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