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

    def merge_default(self, p, dp):
        """Add skipped key from default."""
        key = 'workflow'
        if key in p and len(p[key]) > 0:
            if 'pipeline_id' not in p[key]:
                p[key]['pipeline_id'] = dp[key]['pipeline_id']
            if 'endpoint_id' not in p[key]:
                p[key]['endpoint_id'] = dp[key]['endpoint_id']
            if 'control'  not in p[key]:
                p[key]['control'] = dp[key]['control']
            else:
                for subkey in dp[key]['control']:
                    if subkey not in p[key]['control']:
                        p[key]['control'][subkey] = dp[key]['control'][subkey]
        else:
            p[key] = dp[key]

        for key in ['pipeline', 'endpoint', 'metric', 'gs']:
            if key in p and len(p[key])>0:
                if key == 'gs':
                    # update with default below
                    continue
                for conf in p[key].values():
                    for subkey in dp[key]['default'].keys():
                        if subkey not in conf:
                            conf[subkey] = dp[key]['default'][subkey]
            else:
                p[key] = {'default': dp[key]['default']}

        # this one optional
        # update gs only if default fit endpoint
        for endpoint in p['endpoint'].values():
            if endpoint['fit'] == 'default':
                gs_id = endpoint['gs_id']
                if gs_id in p['gs']:
                    conf = p['gs'][gs_id]
                elif gs_id == 'auto' and len(p['gs'])==1:
                    gs_id = list(p['gs'].keys())[0]
                    conf = p['gs'][gs_id]
                else:
                    # raise error below
                    continue
                for subkey in dp[key]['default'].keys():
                    if subkey not in conf:
                        conf[subkey] = dp[key]['default'][subkey]

        for data_id in p['data']:
            for subkey in dp['data']['default']:
                if subkey not in p['data'][data_id]:
                    p['data'][data_id][subkey] = dp['data']['default'][subkey]


    def parse_params(self, p):
        # check if configuration name is skipped, set under 'user' name
        reserved = {'endpoint': mlshell.default.DEFAULT_PARAMS['endpoint']['default'].keys(),
                    'gs': mlshell.default.DEFAULT_PARAMS['gs']['default'].keys(),
                    'pipeline': mlshell.default.DEFAULT_PARAMS['pipeline']['default'].keys(),}

        for key in ['pipeline', 'endpoint', 'gs']:
            if key in p and len(p[key])>0:
                subkeys = set(p[key].keys())
                if subkeys - reserved[key]:
                    pass
                    # [deprecated] check key names when merge default
                    # for subkey in subkeys:
                    #     subsubkeys = set(p[key][subkey].keys())
                    #     diff = subsubkeys - reserved[key]
                    #     if diff:
                    #         raise KeyError(f"Unknown keys {diff} in {key}__{subkey} conf.py")
                else:
                    p[key]['user'] = p.pop(key)

        if 'metric' in p and len(p['metric']) > 0:
            item = list(p['metric'].items())[0]
            if isinstance(item, dict):
                p['metric']['user'] = p.pop('metric')

        if 'data' not in p or not p['data']:
            raise KeyError('Specify dataset configuration in conf.py.')

        # merge with default parameters
        self.merge_default(p, copy.deepcopy(mlshell.default.DEFAULT_PARAMS))

        pipeline_id = p['workflow']['pipeline_id']
        endpoint_id = p['workflow']['endpoint_id']
        ids = {
            'pipeline_id': pipeline_id,
            'endpoint_id': endpoint_id,
            'gs_id': p['endpoint'][endpoint_id]['fit']['gs_id'] if endpoint_id is not 'auto' else 'auto',
            'metric_id': p['endpoint'][endpoint_id]['validate']['metric_id'] if endpoint_id is not 'auto' else 'auto',
        }
        # if pipeline_id specified and more than one conf in dict,
        # set with pipeline_id.
        if ids['pipeline_id'] is not 'auto':
            for key in ['endpoint', 'gs', 'metric']:
                name_id = f'{key}_id'
                if key in p and len(p[key]) > 1:
                    if ids[name_id] is 'auto':
                        ids[name_id] = ids['pipeline_id']

        # if specified try to find, if 'auto' use single or pipeline_id
        for key in ['pipeline', 'endpoint', 'gs', 'metric']:
            name_id = f'{key}_id'
            if key in p and len(p[key]) > 0:
                if ids[name_id] is not 'auto':
                    if ids[name_id] not in p[key]:
                        raise ValueError(f"Unknown {name_id} configuration: {ids[name_id]}.")
                else:
                    if len(p[key]) == 1:
                        ids[name_id] = list(p[key].keys())[0]
                    else:
                        raise ValueError(f"Multiple {key} configuration provided, specify {name_id}.")
            # [deprecated] after merge 'default' already in
            #else:
            #    if ids['pipeline_id'] is 'auto':
            #        ids[f'{key}_id'] = 'default'
            #    else:
            #        raise ValueError('unknown pipeline_id')


        # find out which data_id to load in current workflow
        data_ids = {value['data_id'] for value in p['endpoint'][ids['endpoint_id']].values()
                    if 'data_id' in value}

        miss_data_ids = set()
        for data_id in data_ids:
            if data_id not in p['data']:
                miss_data_ids.add(data_id)
        if miss_data_ids:
            raise KeyError(f"Unknown {name_id} configuration(s): {miss_data_ids}.")

        # set random state as soon as possible.
        # checked, work for whole process (estimator has its own seed).
        if 'seed' in p:
            rd.seed(p['seed'])
            np.random.seed(p['seed'])

        # [deprecated]
        # # flatten
        # # metrics
        # params_flat = {
        #     'metrics': params.pop('metrics'),
        # }
        # # if user specified through '__'
        # for key in params:
        #     if '__' in key:
        #         params_flat[key] = params.pop(key)

        # def callable_join(lis):
        #     return '__'.join(lis)
        # # data with depth = 2
        # params_data = params.pop('data')
        # dic_flatter(params_data, params_flat, keys_lis_prev=['data'],
        #             key_transform=callable_join, max_depth=2)
        # # else with depth 1
        # dic_flatter(params, params_flat, key_transform=callable_join, max_depth=1)

        res = {
            'workflow': p['workflow'],
            'endpoint': p['endpoint'][ids['endpoint_id']],
            'pipeline': p['pipeline'][ids['pipeline_id']],
            'metric': p['metric'][ids['metric_id']],
            'gs': p['gs'][ids['gs_id']],
            'data': {data_id:p['data'][data_id] for data_id in data_ids},
        }

        return res



if __name__ == '__main__':
    pass
