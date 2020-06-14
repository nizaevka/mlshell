import logging


class Producer(object):
    def __init__(self, project_path='', logger=None):
        self.logger = logger if logger else logging.Logger(__class__.__name__)
        self.project_path = project_path

    def produce(self, init, steps, objects):
        res = init
        for step in steps:
            key = step[0]
            val = step[1] if len(step) > 1 else {}
            if not isinstance(val, dict):
                raise ValueError(f'Value under {key} should be a dictionary.')
            res = getattr(self, key)(res, objects=objects, **val)
        res = self._check(res)
        return res

    def _check(self, res):
        """Additional result check."""
        return res
