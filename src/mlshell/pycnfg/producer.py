"""
The :mod:`pycnfg.produce` includes class to produce configuration object.
Use it as Mixin to add desired endpoints.

Support method to cache/read intermediate state of object (pickle/unpickle).
It useful to save time when reusing a configuration.

"""


import glob
import json
import logging
import os

import dill
import jsbeautifier
import mlshell.pycnfg as pycnfg
import numpy as np
import pandas as pd


class Producer(object):
    """Execute configuration steps.

    Interface: produce, dump_cache, load_cache.

    Parameters
    ----------
    objects : dict
        Dictionary with resulted objects from previous executed producers:
        {'section_id__config__id', object}.
    oid : str
        Unique identifier of produced object.

    Attributes
    ----------
    objects : dict
        Dictionary with resulted objects from previous executed producers:
        {'section_id__config__id', object,}
    oid : str
        Unique identifier of produced object.
    logger : logger object
        Default logger logging.getLogger().
    project_path: None
        Absolute path to project dir pycnfg.find_path().

    """
    _required_parameters = ['objects', 'oid']

    def __init__(self, objects, oid):
        self.objects = objects
        self.logger = logging.getLogger()
        self.project_path = pycnfg.find_path()
        self.oid = oid

    def produce(self, init, steps):
        """Execute configuration steps.

        consecutive:
        init = getattr(self, 'method_id')(init, objects=objects, **kwargs)`

        Parameters
        ----------
        init: object
            Will be passed as arg in each step and get back as result.
        steps : list of tuples ('method_id', {**kwargs})
            List of `self` methods to run consecutive with kwargs.

        Returns
        -------
        configs : list of tuple [('section_id__config__id', config), ...]
            List of configurations, prepared for execution.

        """
        self.logger.info(f"|__ CONFIGURATION: {self.oid}")
        self.logger.debug(f"Used params:\n"
                          f"    {jsbeautifier.beautify(str(steps))}")
        res = init
        for step in steps:
            method = step[0]
            kwargs = step[1] if len(step) > 1 else {}
            if not isinstance(kwargs, dict):
                raise ValueError(f'Value under {method} '
                                 f'should be a dictionary.')
            kwargs = self._object_resolve(kwargs, self.objects)
            res = getattr(self, method)(res, **kwargs)
        # Add identifier.
        res.oid = self.oid
        res = self._check(res)
        return res

    def dump_cache(self, obj, prefix=None, cachedir=None, fformat='pickle',
                   **kwargs):
        """Pickle intermediate object state to disk.

        Parameters
        ----------
        obj : picklable
            Object to dump.
        prefix : str, optional (default=None)
            File identifier, added to filename. If None, 'self.oid' is used.
        cachedir : str, optional(default=None)
            Absolute path dump dir or relative to 'self.project_dir' started
            with './'. Created, if not exists. If None,"self.project_path/temp
            /objects" is used.
        fformat : 'pickle'/'hr', optional default('pickle')
            If 'pickle', dump `obj` via dill lib. If 'hr' try to decompose
            in human-readable csv/json (only for dictionary).
        **kwargs : kwargs
            Additional parameters to pass in .dump().

        Returns
        -------
        obj : picklable
            Unchanged input for compliance with producer logic.

        """
        if not prefix:
            prefix = self.oid
        if not cachedir:
            cachedir = f"{self.project_path}/temp/objects"
        elif cachedir.startswith('./'):
            cachedir = f"{self.project_path}/{cachedir[2:]}"

        # Create temp dir for cache if not exist.
        if not os.path.exists(cachedir):
            os.makedirs(cachedir)
        for filename in glob.glob(f"{cachedir}/{prefix}*"):
            os.remove(filename)
        fps = set()
        if fformat == 'pickle':
            filepath = f'{cachedir}/{prefix}_.dump'
            fps.add(filepath)
            dill.dump(obj, filepath, **kwargs)
        elif fformat == 'hr':
            filepaths = self._hr_dump(obj, cachedir, prefix, **kwargs)
            fps.add(filepaths)
        else:
            raise ValueError(f"Unknown 'fformat' {fformat}.")

        self.logger.warning('Warning: update cache file(s):\n'
                            '    {}'.format('\n    '.join(fps)))
        return obj

    def load_cache(self, obj, prefix=None, cachedir=None, fformat='pickle',
                   **kwargs):
        """Load intermediate object state from disk.

        Parameters
        ----------
        obj : picklable
            Object template, will be updated for 'hr', ignored for 'pickle'.
        prefix : str, optional (default=None)
            File identifier. If None, 'self.oid' is used.
        fformat : 'pickle'/'hr', optional default('pickle')
            If 'pickle', load object via dill lib.
            If 'hr' try to compose csv/json files in a dictionary.
        cachedir : str, optional(default=None)
            Absolute path load dir or relative to 'self.project_dir' started
            with './'. If None,"self.project_path/temp/objects" is used.
        **kwargs : kwargs
            Additional parameters to pass in .load().

        Returns
        -------
        obj : picklable object
            Loaded cache.

        """
        if not prefix:
            prefix = self.oid
        if not cachedir:
            cachedir = f"{self.project_path}/temp/objects"
        elif cachedir.startswith('./'):
            cachedir = f"{self.project_path}/{cachedir[2:]}"

        if fformat == 'pickle':
            filepath = f'{cachedir}/{prefix}_.dump'
            obj = dill.load(filepath, **kwargs)
        elif fformat == 'hr':
            ob = self._hr_load(cachedir, prefix, **kwargs)
            obj.update(ob)
        else:
            raise ValueError(f"Unknown 'fformat' {fformat}.")
        self.logger.warning(f"Warning: use cache file(s):\n    {cachedir}")
        return obj

    def _object_resolve(self, kwargs, objects):
        """Substitute objects in kwargs.

        Checks if val is str/list of str and looks up in object keys (no'_id').
        """
        for key, val in kwargs:
            if not key.endswith('_id'):
                if isinstance(val, list):
                    names = [f"{key.replace('_id', '')}__{v}" for v in val
                             if isinstance(v, str)]
                    kwargs[key] = [objects[name] for name in names
                                   if name in objects]
                else:
                    if not isinstance(val, str):
                        continue
                    name = f"{key.replace('_id', '')}__{val}"
                    kwargs[key] = objects[name]
        return kwargs

    def _check(self, res):
        """Additional result check."""
        return res

    def _hr_dump(self, obj, filedir, prefix, **kwargs):
        """Dump an dictionary to a file(s) in human-readable format.

        Traverse dictionary items and dump pandas/numpy object to separate
        csv-files, others to json-files.

        Parameters
        ----------
        obj : dict
            Object to dump.
        filedir : str
            Dump directory.
        prefix : str
            Prefix for files names.
        **kwargs : dict {'json':kwargs, 'csv':kwargs}
            Additional parameters to pass in low-level functions.

        Returns
        -------
        filenames : set of str
            Resulted filenames "prefix_key_.ext".

        """
        if not isinstance(obj, dict):
            raise ValueError("Object should be a dictionary.")
        filenames = set()
        for key, val in obj.items():
            if isinstance(obj[key], (pd.DataFrame, pd.Series)):
                filepath = f'{filedir}/{prefix}_{key}_.csv'
                filenames.add(filepath)
                with open(filepath, 'w', newline='') as f:
                    val.to_csv(f, mode='w', header=True,
                               index=True, line_terminator='\n',
                               **kwargs['csv'])
            elif isinstance(obj[key], np.ndarray):
                filepath = f'{filedir}/{prefix}_{key}_.csv'
                filenames.add(filepath)
                with open(filepath, 'w', newline='') as f:
                    pd.DataFrame(val).to_csv(f, mode='w', header=True,
                                             index=True, line_terminator='\n',
                                             **kwargs['csv'])
                # [alternative] np.savetxt(filepath, val, delimiter=",")
            else:
                filepath = f'{filedir}/{prefix}_{key}_.json'
                filenames.add(filepath)
                with open(filepath, 'w') as f:
                    # items() preserve first level dic keys as int.
                    json.dump(list(val.items()), f, **kwargs['json'])
        return filenames

    def _hr_load(self, filedir, prefix, **kwargs):
        """Load an object from file(s) and compose in dictionary.

        Parameters
        ----------
        filedir : str
            Load directory.
        prefix : str
            Prefix for target files names.
        **kwargs : dict {'json':kwargs, 'csv':kwargs}
            Additional parameters to pass in low-level functions.

        Returns
        -------
        obj : dict
            Resulted object.

        Notes
        -----
        Dictionary keys are gotten from filenames "prefix_key_.ext".

        """
        obj = {}
        for filepath in glob.glob(f"{filedir}/{prefix}*"):
            key = '_'.join(filepath.split('_')[1:-1])
            if filepath.endswith('.csv'):
                with open(filepath, 'r') as f:
                    obj[key] = pd.read_csv(f, sep=",",
                                           index_col=0, **kwargs['csv'])
            else:
                with open(filepath, 'r') as f:
                    # [alternative] object_hook=json_keys2int)
                    obj[key] = dict(json.load(f, **kwargs['json']))
        return obj


if __name__ == '__main__':
    pass
