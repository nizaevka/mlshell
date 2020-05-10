"""Module contains class to read configuration file from project directory."""


import importlib.util
import sys
import logging
from mlshell.callbacks import dic_flatter
from mlshell.libs import copy, np, rd
import mlshell.default


class GetParams(object):
    """Class to read workflow configuration from file"""
    def __init__(self, logger=None):
        if logger is None:
            self.logger = logging.Logger('GetParams')
        else:
            self.logger = logger
        self.logger = logger

    def get_params(self, project_path, params=None):
        self.logger.info("\u25CF READ CONFIGURATION")
        if params is None:
            dir_name = project_path.split('/')[-1]
            spec = importlib.util.spec_from_file_location(f'conf', f"{project_path}/conf.py")
            conf = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(conf)
            sys.modules['conf'] = conf  # otherwise problem with pickle, depends on the module path
            params = copy.deepcopy(conf.params)
        return self.parse_params(params)

    def parse_params(self, p):
        # [deprecated] unreliable for custom, always contain name
        # # check if configuration name is skipped, set under 'user' name
        # reserved = {'endpoint': mlshell.default.DEFAULT_PARAMS['endpoint']['default'].keys(),
        #             'gs': mlshell.default.DEFAULT_PARAMS['gs']['default'].keys(),
        #             'pipeline': mlshell.default.DEFAULT_PARAMS['pipeline']['default'].keys(),}
        # for key in ['pipeline', 'endpoint', 'gs']:
        #     if key in p and len(p[key])>0:
        #         subkeys = set(p[key].keys())
        #         if subkeys - reserved[key]:
        #             pass
        #             # [deprecated] check key names when merge default
        #             # for subkey in subkeys:
        #             #     subsubkeys = set(p[key][subkey].keys())
        #             #     diff = subsubkeys - reserved[key]
        #             #     if diff:
        #             #         raise KeyError(f"Unknown keys {diff} in {key}__{subkey} conf.py")
        #         else:
        #             p[key]['user'] = p.pop(key)
        # if 'metric' in p and len(p['metric']) > 0:
        #     item = list(p['metric'].items())[0]
        #     if isinstance(item, dict):
        #         p['metric']['user'] = p.pop('metric')
        if 'data' not in p or not p['data']:
            raise KeyError('Specify dataset configuration in conf.py.')

        # merge with default parameters
        self.merge_default(p, copy.deepcopy(mlshell.default.DEFAULT_PARAMS))

        if not p['workflow']['endpoint_id']:
            if len(p['endpoint']) > 1:
                raise ValueError(f"Multiple 'endpoint' configuration provided, specify 'endpoint' id in 'workflow'.")
            else:
                p['workflow']['endpoint_id'] = list(p['endpoint'].keys())[0]
        endpoint_ids = p['workflow']['endpoint_id']

        if isinstance(endpoint_ids, str):
            endpoint_ids = [endpoint_ids]
        ids = {'endpoint': endpoint_ids}
        for endpoint_id in endpoint_ids:
            self.resolve_none(p, endpoint_id, ids)

        # remain only used configuration
        res = {'workflow': p['workflow'], }
        for key, val in ids.items():
            # if not empty configuration
            if val:
                res[key] = {id_: p[key][id_] for id_ in val}
        # [deprecated] too complicated
        # # if pipeline_id specified and more than one conf in dict,
        # # set with pipeline_id.
        # if ids['pipeline_id'] is not 'auto':
        #     for key in ['endpoint', 'gs', 'metric']:
        #         name_id = f'{key}_id'
        #         if key in p and len(p[key]) > 1:
        #             if ids[name_id] is 'auto':
        #                 ids[name_id] = ids['pipeline_id']

        # # if specified try to find, if 'auto' use single or pipeline_id
        # for key in ['pipeline', 'endpoint', 'gs', 'metric']:
        #     name_id = f'{key}_id'
        #     if key in p and len(p[key]) > 0:
        #         if ids[name_id] is not 'auto':
        #             if ids[name_id] not in p[key]:
        #                 raise ValueError(f"Unknown {name_id} configuration: {ids[name_id]}.")
        #         else:
        #             if len(p[key]) == 1:
        #                 ids[name_id] = list(p[key].keys())[0]
        #             else:
        #                 raise ValueError(f"Multiple {key} configuration provided, specify {name_id}.")
        #     # [deprecated] after merge 'default' already in
        #     #else:
        #     #    if ids['pipeline_id'] is 'auto':
        #     #        ids[f'{key}_id'] = 'default'
        #     #    else:
        #     #        raise ValueError('unknown pipeline_id')


        # miss_data_ids = set()
        # for data_id in data_ids:
        #     if data_id not in p['data']:
        #         miss_data_ids.add(data_id)
        # if miss_data_ids:
        #     raise KeyError(f"Unknown {name_id} configuration(s): {miss_data_ids}.")

        # [deprecated] TODO: should be enpdoint-wise, also at each subfunction,
        #                   ny default disabled.
        # set global random state as soon as possible.
        # checked, work for whole process (estimator has its own seed).
        # if 'seed' in p['endpoint'][endpoint_id]['global']:
        #     seed = p['endpoint'][endpoint_id]['global']['seed']
        #     rd.seed(seed)
        #     np.random.seed(seed)

        return res

    def resolve_none(self, p, endpoint_id, ids):
        # if some None, use global, if global None, use from conf list, if more than one, error

        if 'global' not in p['endpoint'][endpoint_id]:
            p['endpoint'][endpoint_id]['global'] = {}

        # metric by default all
        if 'metric' not in p['endpoint'][endpoint_id]['global'] or not p['endpoint'][endpoint_id]['global']['metric']:
            p['endpoint'][endpoint_id]['global']['metric'] = p['metric'].keys()

        # keys with not separate conf (like 'seed', global None is possible)
        primitive = {key for key in p['endpoint'][endpoint_id]['global'] if key not in p}
        for key in primitive:
            key_id = key
            # read glob val if exist
            if key_id in p['endpoint'][endpoint_id]['global']:
                glob_val = p['endpoint'][endpoint_id]['global'][key_id]
            else:
                glob_val = None
            for subkey, value in p['endpoint'][endpoint_id].items():
                if subkey is not 'global' and key_id in value:
                    # if None use global
                    if not value[key_id]:
                        value[key_id] = glob_val

        # keys with separate conf (like 'data', 'pipeline', 'metric', 'gs')
        nonprimitive = {key for key in p.keys() if key not in ['workflow', 'endpoint']}
        for key in nonprimitive:
            key_id = key
            # read glob val if exist
            if key_id in p['endpoint'][endpoint_id]['global']:
                glob_val = p['endpoint'][endpoint_id]['global'][key_id]
            else:
                glob_val = None
            if key not in ids:
                ids[key] = set()
            for subkey, value in p['endpoint'][endpoint_id].items():
                if subkey is not 'global' and key_id in value:
                    # if None use global
                    if not value[key_id]:
                        # if global None use from conf (if only one provided)
                        # for metrics not None is guaranteed before
                        if not glob_val:
                            if len(p[key]) > 1:
                                raise ValueError(
                                    f"Multiple {key} configurations provided, specify key:\n"
                                    f"    'endpoint:{endpoint_id}:{subkey}:{key_id}' or 'endpoint:{endpoint_id}:global:{key_id}'.")
                            else:
                                glob_val = list(p[key].keys())[0]
                        value[key_id] = glob_val
                    # check if conf available
                    if not isinstance(value[key_id], str) and hasattr(value[key_id], '__iter__'):
                        # for compatibility with sequence of ids (like metric)
                        confs = value[key_id]
                    else:
                        confs = [value[key_id]]
                    for conf in confs:
                        if conf not in p[key]:
                            raise ValueError(f"Unknown {key} configuration: {conf}.")
                    # set
                    ids[key].update(confs)

    def merge_default(self, p, dp):
        """Add skipped key from default."""
        self.check_params_keys(p, dp)

        key = 'workflow'
        if key in p and len(p[key]) > 0:
            # [deprecated]
            # if 'pipeline_id' not in p[key]:
            #     p[key]['pipeline_id'] = dp[key]['pipeline_id']
            if 'endpoint_id' not in p[key] or not p[key]['endpoint_id']:
                p[key]['endpoint_id'] = copy.deepcopy(dp[key]['endpoint_id'])
            if 'steps' not in p[key] or p[key] is None:
                p[key]['steps'] = copy.deepcopy(dp[key]['steps'])
            # [deprecated] use user steps without additions
            # else:
            #     for subkey in dp[key]['steps']:
            #         if subkey not in p[key]['steps']:
            #             p[key]['steps'][subkey] = dp[key]['steps'][subkey]
        else:
            p[key] = copy.deepcopy(dp[key])

        for key in ['pipeline', 'endpoint']:
            if key in p and len(p[key]) > 0:
                for conf in p[key].values():
                    for subkey in dp[key]['default'].keys():
                        if subkey not in conf:
                            conf[subkey] = copy.deepcopy(dp[key]['default'][subkey])
            else:
                p[key] = {'default': copy.deepcopy(dp[key]['default'])}

        key = 'metric'
        if key not in p:
            name = p['pipeline']['type']
            p[key] = copy.deepcopy(dp[key][name])

        key = 'gs'
        if key not in p:
            p[key] = {'default': copy.deepcopy(dp[key]['default'])}
        # [deprecated] too complicated, mode to fit method with warning
        # # update gs only if default fit endpoint and explicit name
        # # from endpoints get all gs_ids with default fit func
        # gs_ids = set()
        # for endpoint in p['endpoint'].values():
        #     if endpoint['fit']['func'] is None:
        #         gs_ids.add(endpoint['gs_id'])
        #         gs_ids.add(endpoint['fit']['gs_id'])
        # for gs_id in gs_ids:
        #     if gs_id is not None and gs_id in p['gs']:
        #         for subkey in dp[key]['default'].keys():
        #             if subkey not in conf:
        #                 conf[subkey] = dp[key]['default'][subkey]
        # # double gs_id extraction, here and below
        # # this one optional
        # for endpoint in p['endpoint'].values():
        #     if endpoint['fit']['func'] is None:
        #         gs_id = endpoint['gs_id']
        #         if gs_id in p['gs']:
        #             conf = p['gs'][gs_id]
        #         elif gs_id is None and len(p['gs']) == 1:
        #             gs_id = list(p['gs'].keys())[0]
        #             conf = p['gs'][gs_id]
        #         else:
        #             # raise error below
        #             continue
        #         for subkey in dp[key]['default'].keys():
        #             if subkey not in conf:
        #                 conf[subkey] = dp[key]['default'][subkey]

        # [deprecated] always user-defined
        # 'data'
        # for data_id in p['data']:
        #     for subkey in dp['data']['default']:
        #         if subkey not in p['data'][data_id]:
        #             p['data'][data_id][subkey] = dp['data']['default'][subkey]
        return None

    def check_params_keys(self, p, dp):
        miss_keys = set()
        for key in list(p.keys()):
            if key not in dp:
                miss_keys.add(key)
                # del p[key]
        if miss_keys:
            # user can create configuration for arbitrary param
            # check if dict type
            for key in miss_keys:
                if not isinstance(p[key], dict):
                    raise TypeError(f"Custom params[{key}] should be the dict instance.")
            # self.logger.warning(f"Ignore unknown key(s) in conf.py params, check\n    {miss_keys}")


if __name__ == '__main__':
    pass
