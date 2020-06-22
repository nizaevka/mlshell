def _dic_flatter(dic, dic_flat, key_transform=None, val_transform=None,
                keys_lis_prev=None, max_depth=None):
    """Flatten the dict.

    {'a':{'b':[], 'c':[]},} =>  {('a','b'):[], ('a','c'):[]}

    Args:
        dic (dict): input dictionary.
        dic_flat (dict): result dictionary.
        key_transform (callback, optional (default=None)): apply to end keys,
            if None '__'.join() is used.
        val_transform (callback, optional (default=None)): apply to end dic values,
            if None identity function f(x)=x is used.
        keys_lis_prev: need for recursion.
        max_depth (None or int, optional (default=None)): depth of recursion.

    Note:
        if dict is empty, stop.

    """
    if val_transform is None:
        def id_func(x): return x
        val_transform = id_func
    if key_transform is None:
        key_transform = tuple

    if keys_lis_prev is None:
        keys_lis_prev = []
    for key, val in dic.items():
        keys_lis = keys_lis_prev[:]
        keys_lis.append(key)
        if isinstance(val, dict) and val and (len(keys_lis_prev) < max_depth or not max_depth):
            _dic_flatter(val, dic_flat,
                        key_transform=key_transform, val_transform=val_transform,
                        keys_lis_prev=keys_lis, max_depth=max_depth)
        else:
            if len(keys_lis) == 1:
                dic_flat[keys_lis[0]] = val_transform(val)
            else:
                dic_flat[key_transform(keys_lis)] = val_transform(val)


def _json_keys2int(x):
    # dic key to int
    if isinstance(x, dict) and 'categoric' not in x:
        return {int(k): v for k, v in x.items()}
    return x