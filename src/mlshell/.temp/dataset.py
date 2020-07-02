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