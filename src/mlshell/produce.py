"""
The module includes class to execute configuration.
"""


class Producer(object):
    """Execute configuration steps.

    Interface: produce.
    Use class as Mixin to add desired methods.

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

    def produce(self, init, steps, objects):
        """Execute configuration steps.

        consecutive:
        init = getattr(self, 'method_id')(init, objects=objects, **kwargs)`

        Parameters
        ----------
        init: object
            Will be passed as arg in each step and get back as result.

        steps : list of tuples ('method_id', {**kwargs})
            List of `self` methods to run consecutive with kwargs.

        objects : dict {'section_id__config__id', object,}
            Dictionary with resulted objects from previous `configs` execution.

        Returns
        -------
        configs : list of tuple [('section_id__config__id', config), ...]
            List of configurations, prepared for execution.

        """
        res = init
        for step in steps:
            method = step[0]
            kwargs = step[1] if len(step) > 1 else {}
            if not isinstance(kwargs, dict):
                raise ValueError(f'Value under {method} '
                                 f'should be a dictionary.')
            kwargs = self._fill_object(kwargs, objects)
            res = getattr(self, method)(res, objects=objects, **kwargs)
        res = self._check(res)
        return res

    def _fill_object(self, kwargs, objects):
        """Substitute objects in kwargs if exist.

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
