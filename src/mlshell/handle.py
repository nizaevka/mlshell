"""Module contains class to read configuration file from project directory."""


import importlib.util
import sys
import logging
from mlshell.libs import copy, np, rd, heapq, inspect, collections
import mlshell.default
import types


class ConfHandler(object):
    """Class to read workflow configuration from file"""
    def __init__(self, project_path='', logger=None):
        self.logger = logger if logger else logging.Logger(__class__.__name__)
        self.project_path = project_path

    def read(self, conf=None, default_conf=None):
        self.logger.info("\u25CF READ CONFIGURATION")
        if conf is None:
            dir_name = self.project_path.split('/')[-1]
            spec = importlib.util.spec_from_file_location(f'conf', f"{self.project_path}/conf.py")
            conf_file = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(conf_file)
            sys.modules['conf'] = conf_file  # otherwise problem with pickle, depends on the module path
            conf = copy.deepcopy(conf_file.conf)
        return self._parse_conf(conf, default_conf)

    def exec(self, configs):
        objects = {}
        for config in configs:
            name, val = config
            self.logger.info(f"\u25CF HANDLE {name}")
            self.logger.info(f"Configuration:\n    {name}")
            objects[name] = self._exec(val, objects)
        return objects

    def _parse_conf(self, p, dp=None):
        if dp is None:
            dp = copy.deepcopy(mlshell.default.DEFAULT_PARAMS)
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
        if 'dataset' not in p or not p['dataset']:
            raise KeyError('Specify dataset configuration in conf.py.')

        # merge with default parameters

        self.merge_default(p, dp)

        # Resolve endpoint.
        if p['workflow']['endpoint_id'] is None:
            if len(p['endpoint']) > 1:
                raise ValueError(f"Multiple 'endpoint' configuration provided,"
                                 f" specify 'endpoint' id in 'workflow'.")
            else:
                p['workflow']['endpoint_id'] = list(p['endpoint'].keys())[0]
        endpoint_ids = p['workflow']['endpoint_id']
        if isinstance(endpoint_ids, str):
            endpoint_ids = [endpoint_ids]
        ids = {'endpoint': endpoint_ids}

        # Resolve parameters in endpoint(s).
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
        #     if data_id not in p['dataset']:
        #         miss_data_ids.add(data_id)
        # if miss_data_ids:
        #     raise KeyError(f"Unknown {name_id} configuration(s): {miss_data_ids}.")

        # [deprecated] global seed is bad practice, anyway available in conf.
        # set global random state as soon as possible.
        # checked, work for whole process (estimator has its own seed).
        # if 'seed' in p['endpoint'][endpoint_id]['global']:
        #     seed = p['endpoint'][endpoint_id]['global']['seed']
        #     rd.seed(seed)
        #     np.random.seed(seed)

        res = self._priority_arrange(res)
        # [('config__id', config), ...]
        # check repeated name
        non_uniq = [k for (k, v) in
                    collections.Counter(list(zip(*res))[0]).items() if v > 1]
        if non_uniq:
            raise ValueError(f"Non-unique configuration id found:\n"
                             f"    {non_uniq}")
        return res

    def resolve_none(self, p, endpoint_id, ids):
        """Auto resolution for None parameters in endpoint section.

        If parameter None => use global,
        if global None => use from available configuration,
        if more than one configuration => ValueError.
        If parameter name contain '_id', None substitute with conf name,
        else with conf itself.
        if no configuration => remain None.

        """
        # TODO: check consequence.
        # [deprecated] already exist in default.
        # if 'global' not in p['endpoint'][endpoint_id]:
        #     p['endpoint'][endpoint_id]['global'] = {}

        # TODO: check consequence.
        # [deprecated] asymmetry + should be user defined.
        # metric by default all
        # if 'metric' not in p['endpoint'][endpoint_id]['global'] \
        #         or not p['endpoint'][endpoint_id]['global']['metric']:
        #     p['endpoint'][endpoint_id]['global']['metric'] = p['metric'].keys()

        # Keys resolved via global.
        # (exist in global and no separate conf).
        primitive = {key for key in p['endpoint'][endpoint_id]['global']
                     if key.replace('_id', '') not in p}
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

        # Keys resolved via separate conf section.
        # two separate check: contain '_id' or not.
        nonprimitive = {key for key in p.keys() if key not in ['workflow', 'endpoint']}
        for key in nonprimitive:
            # read glob val if exist
            if key in p['endpoint'][endpoint_id]['global']:
                glob_val = p['endpoint'][endpoint_id]['global'][key]
            elif f"{key}_id" in p['endpoint'][endpoint_id]['global']:
                glob_val = p['endpoint'][endpoint_id]['global'][f"{key}_id"]
            else:
                glob_val = None
            if key not in ids:
                ids[key] = set()
            for subkey, value in p['endpoint'][endpoint_id].items():
                if subkey is 'global':
                    continue
                key_id = None
                if key in value:
                    key_id = key
                elif f"{key}_id" in value:
                    key_id = f"{key}_id"
                if key_id:
                    # If None use global.
                    if not value[key_id]:
                        # If global None use from conf (if only one provided)
                        # for metrics not None is guaranteed before.
                        if not glob_val:
                            if len(p[key]) > 1:
                                raise ValueError(
                                    f"Multiple {key} configurations provided, specify key:\n"
                                    f"    'endpoint:{endpoint_id}:{subkey}:{key_id}' or 'endpoint:{endpoint_id}:global:{key_id}'.")
                            else:
                                glob_val = list(p[key].keys())[0]
                        value[key_id] = glob_val

                    # Check if conf available.
                    if not isinstance(value[key_id], str) and hasattr(value[key_id], '__iter__'):
                        # for compatibility with sequence of ids (like metric)
                        confs = value[key_id]
                    else:
                        confs = [value[key_id]]
                    for conf in confs:
                        if conf not in p[key]:
                            raise ValueError(f"Unknown {key} configuration: {conf}.")

                    # Substitute either id(s), or conf.
                    if key_id.endswith('_id'):
                        ids[key].update(confs)
                    else:
                        # Set inplace with `init` copy (contain mutable).
                        # [alternative] not copy, so will always contain fresh 'object'
                        #     currently only template is copy => for kw_args without factory
                        #     kwargs with factory better via separate storage.
                        #     It is also possible to resolve `init`/`storage` to skip `objects`
                        #     but in that case we need to fix structure => less flexible.
                        if len(confs) > 1:
                            value[key_id] = copy.deepcopy([p[key][conf]['init'] for conf in confs])
                        else:
                            value[key_id] = copy.deepcopy(p[key][confs[0]]['init'])

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

        for key in ['endpoint']:  # [deprecated] 'pipeline', unified
            if key in p and len(p[key]) > 0:
                # Use default subkeys only for default class.
                if 'class' not in p[key] or p[key]['class'] is None:
                    for conf in p[key].values():
                        for subkey in dp[key]['default'].keys():
                            if subkey not in conf:
                                conf[subkey] = copy.deepcopy(dp[key]['default'][subkey])
            else:
                p[key] = {'default': copy.deepcopy(dp[key]['default'])}

        # TODO: all others are customs, do similar as for endpoint.

        # TODO: better apply in validator factory.
        key = 'metric'
        if key not in p:
            name = p['pipeline']['type']
            p[key] = copy.deepcopy(dp[key][name])

        # [deprecated] custom no need in defaults.
        # key = 'gs_params'
        # if key not in p:
        #     p[key] = {'default': copy.deepcopy(dp[key]['default'])}

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
        # 'dataset'
        # for data_id in p['dataset']:
        #     for subkey in dp['dataset']['default']:
        #         if subkey not in p['dataset'][data_id]:
        #             p['dataset'][data_id][subkey] = dp['dataset']['default'][subkey]
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

    def _priority_arrange(self, res):
        min_heap = []
        for key in res:
            for subkey in res[key]:
                val = res[key][subkey]
                # [beta]
                # name = f'{key}__{subkey}'
                name = subkey
                priority = val.get('priority', 0)
                heapq.heappush(min_heap, (priority, (name, val)))
        sorted_ = heapq.nsmallest(len(min_heap), min_heap)
        return list(zip(*sorted_))[1]

    def _exec(self, conf, objects):
        init = conf.get('init', {})
        steps = conf.get('steps', [])
        producer = conf.get('producer', mlshell.Producer)
        patch = conf.get('patch', {})
        if inspect.isclass(init):
            init = init()
        if inspect.isclass(producer):
            producer = producer(project_path=self.project_path, logger=self.logger)
        producer = self._patch(patch, producer)
        return producer.produce(init, steps, objects)

    def _patch(self, patch, producer):
        """Monkey-patching producer.

        producer : class object.
            Object to patch.
        patch : dict {'method_id' : function/existed 'method_id' }.
            Functions to add/rewrite.

        """
        needs_resolve = []
        # update/add new methods
        for key, val in patch.items():
            if isinstance(val, str):
                needs_resolve.append((key, val))
                patch[key] = producer.__getattribute__(val)
            setattr(producer, key, types.MethodType(val, producer))
        # resolve str name for existed methods
        for key, name in needs_resolve:
            setattr(producer, key, getattr(producer, name))
        return producer


if __name__ == '__main__':
    pass
