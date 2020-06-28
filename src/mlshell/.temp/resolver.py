def _extract_ind_name(self, dataset):
    """Extract categoric/numeric names and index."""
    data = dataset.data
    meta = dataset.meta
    categoric_ind_name = {}
    numeric_ind_name = {}
    count = 0
    for ind, column_name in enumerate(data):
        if column_name in meta['targets']:
            count += 1
            continue
        if column_name in meta['categor_features']:
            # Loose categories names.
            categoric_ind_name[ind - count] = \
                (column_name, np.unique(data[column_name]))
        else:
            numeric_ind_name[ind - count] = (column_name,)
    return data, categoric_ind_name, numeric_ind_name