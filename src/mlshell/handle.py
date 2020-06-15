"""The module contains class to read and execute configuration(s).

Configuration is a python dictionary.
Configuration could be passed to `mlshell.run` function or user-defined handler,
where `mlshell.ConfHandler` built objects and executes its endpoints.
Configuration support multiple sections. Each section specify set of
sub-configurations.
Each sub-configuration provide steps to construct an object, that can be
utilize as argument in some other sections.

For each section there is common logic:
{'section_id':
    'configuration_id 1': {
        'init': initial object of custom type.
        'producer': factory class, which contain methods to run steps.
        'patch': add custom method to class.
        'steps': [
            ('method_id 1', kwargs_1),
            ('method_id 2', kwargs_2),
        ],
        'global': shortcut to common parameters.
        'priority': execute priority (integer non-negative number).
    }
    'configuration_id 2':{
        ...
    }
}

The target for each sub-configuration is to create an instance.
`init` is the template for future object (empty dict() for example).
`producer` work as factory, it should contain .produce() method that:
* takes `init` and consecutive pass it to `steps` with specified kwargs.
* return resulting object, that can be used as kwargs for any step in others
sections.
To specify the order in which sections handled, 'priority' key is available.

For flexibility, it is possible to:
* Monkey patch `producer` object with custom functions via `patch` key.
* Specify global value for common kwargs in steps via `global` key.
all keys in configuration (except ['init', 'producer', 'global', 'patch',
'steps', 'priority']) are moved to `global` automatically.
* Create separate section for arbitrary parameter in steps.
So it is sufficient to use `section_id__configuration_id` as kwarg value, then
in step execution kwarg could be gotten from `objects`.
As alternative, if add '_id' postfix to `kwarg_id`,
`init` object from `configuration_id` will be copy on parse

Then there are two ways to define kwarg {`kwarg_id`: kwarg_val}:
* `producer`/`init` can be both class or object.


TODO:
    encompass all sklearn-wise in mlshell.utills.sklearn

TODO [beta]
    * support 'section_id__conf_id', not sure if possible, require read variable name.

Parameters
----------
init : object.
    Initial state for constructed object. Will be passed consecutive in steps
    as argument. If None or skipped, dict() is used.

producer : class or instance.
    Factory to construct an object: `producer.produce(`init`, `steps`, `objects`)`
    will be called, where `objects` is dictionary with previously created
    objects {'section_id__configuration_id': object}.
    If None or skipped, mlshell.Producer is used. If set as class will be
    auto initialized: `producer(project_path, logger)` will be called.

patch : dict {'method_id' : function}.
    Monkey-patching `producer` object with custom functions.

steps : list of tuples (str: 'method_id', Dict: kwargs).
    List of class methods to run consecutive with kwargs.
    Each step should be a tuple: `('method_id', {kwargs to use})`,
    where 'method_id' should match to `producer` functions' names.
    It is possible to omit kwargs, in that case each step executed with kwargs
    set default in corresponding producer method (see producer interface)

    **kwargs : dict {'kwarg_name': value, ...}.
        Arguments depends on workflow methods. It is possible to create
        separate configuration section for any argument and specify the `value`
        either as 'configuration_id', or as list of 'configuration_id'.
        If `value` is set to None, parser try to resolve it. First it searches
        for value in `global` subsection. Then resolver looks up 'kwarg_name'
        in section names. If such section exist, there are two possibilities:
        if `kwarg_name` contain '_id' postfix, resolver substitutes None with
        available `configuration_id`, else without postfix
        resolver substitutes None with copy of configuration `init` object.
        If fails to find resolution, `value` is remained None. In case of
        resulation plurality, ValueError is raised.

global : dict {'kwarg_name': value, ...}.
    Specify values to resolve None for arbitrary kwargs. This is convenient for
    example when we use the same `pipeline` in all methods. It is not rewrite
    not-None values.

priority : non-negative integer, optional (default=0).
    Priority of configuratuon execution. The more the higher priority.
    For two conf with same priority order is not guaranteed.

**keys : arbitraty objects.
    All keys in configuration (except ['init', 'producer', 'global', 'patch',
    'steps', 'priority']) are moved to `global` automatically.

Examples
--------
# Patch producer with custom functions.
def my_func(self, pipeline, dataset):
    # ... custom logic ...
    return

{'patch': {'extra': my_func,},}

"""


import importlib.util
import sys
import logging
from mlshell.libs import copy, np, rd, heapq, inspect, collections
import mlshell.default
import types


class ConfHandler(object):
    """Read and execute configurations.

    Important members are read, exec.

    `mlshell.ConfHandler`:
* Parse section one by one in priority.
* For each configuration in sections:
    * call `producer`.produce(`init`, `steps`, `objects`).
    * store result in built-in `objects` storage under `section_id__configuration_id`.

    Parameters
    ----------
    project_path: str.
        Absolute path to current project dir (with conf.py).
    logger : logger object.
        Logs.

    See Also
    ---------
    :class:`mlshell.Producer`:
        Execute configuration steps.

    """
    _required_parameters = []

    def __init__(self, project_path, logger, **kwargs):
        self.logger = logger
        self.project_path = project_path

    def read(self, conf=None, default_conf=None):
        """Read raw configuration and transform to executable.

        Parse and resolve skipped parameters.

        Parameters
        ----------
        conf : dict or None.
            Set of configurations {'section_id': {'configuration_id': configuration,},}.
            If None, try to read `conf` from `project_path/conf.py`.
        default_conf : dict .
            Set of default configurations. {'section_id': {'configuration_id': configuration, },}
            If None, read from `mlshell.DEFAULT_PARAMS`.

        Notes
        -----
        If section is skipped, default section is used.
        If sub-keys are skipped, default values are used for these sub-keys.
        So in most cases it is enough just to specify 'global'.

        TODO: auto-resolution rules move here.

        """
        self.logger.info("\u25CF READ CONFIGURATION")
        if conf is None:
            dir_name = self.project_path.split('/')[-1]
            spec = importlib.util.spec_from_file_location(f'conf', f"{self.project_path}/conf.py")
            conf_file = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(conf_file)
            # Otherwise problem with pickle, depends on the module path.
            sys.modules['conf'] = conf_file
            conf = copy.deepcopy(conf_file.conf)
        if default_conf is None:
            default_conf = copy.deepcopy(mlshell.DEFAULT_PARAMS)
        return self._parse_conf(conf, default_conf)

    def exec(self, configs):
        objects = {}
        for config in configs:
            name, val = config
            self.logger.info(f"\u25CF HANDLE {name}")
            self.logger.info(f"Configuration:\n    {name}")
            objects[name] = self._exec(val, objects)
        return objects

    def _parse_conf(self, p, dp):
        # Apply default.
        p = self._merge_default(p, dp)
        # Resolve None for configuration.
        ids = {}
        for section_id in p:
            for conf_id in p[section_id]:
                self._resolve_none(p, section_id, conf_id, ids)

        # Remain only used configuration.
        res = {}
        for key, val in ids.items():
            # If not empty configuration.
            if val:
                res[key] = {id_: p[key][id_] for id_ in val}

        res = self._priority_arrange(res)  # [('config__id', config), ...]
        self._check_res(res)
        return res

    def _resolve_none(self, p, section_id, conf_id, ids):
        """Auto resolution for None parameters in endpoint section.

        If parameter None => use global,
        if global None => use from available configuration,
        if more than one configuration => ValueError.
        If parameter name contain '_id', None substitute with conf name,
        else with conf itself.
        if no configuration => remain None.

        [alternative] update with global when call step.

        """
        # Assemble unknown keys to global.
        for key in p[section_id][conf_id]:
            if key not in ['init', 'producer', 'global',
                           'patch', 'steps', 'priority']:
                p[section_id][conf_id]['global'].update({key: p[section_id][conf_id].pop(key)})

        # Keys resolved via global.
        # (exist in global and no separate conf).
        primitive = {key for key in p[section_id][conf_id]['global']
                     if key.replace('_id', '') not in p}
        for key in primitive:
            key_id = key
            # read glob val if exist
            if key_id in p[section_id][conf_id]['global']:
                glob_val = p[section_id][conf_id]['global'][key_id]
            else:
                glob_val = None
            for step in p[section_id][conf_id]['steps']:
                if len(step) <= 1:
                    continue
                subkey, value = step
                if key_id in value:
                    # if None use global
                    if not value[key_id]:
                        value[key_id] = glob_val

        # Keys resolved via separate conf section.
        # two separate check: contain '_id' or not.
        nonprimitive = {key for key in p.keys()}
        for key in nonprimitive:
            # read glob val if exist
            if key in p[section_id][conf_id]['global']:
                glob_val = p[section_id][conf_id]['global'][key]
            elif f"{key}_id" in p[section_id][conf_id]['global']:
                glob_val = p[section_id][conf_id]['global'][f"{key}_id"]
            else:
                glob_val = None
            if key not in ids:
                ids[key] = set()
            for step in p[section_id][conf_id]['steps']:
                if len(step) <= 1:
                    continue
                subkey, value = step
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
                                    f"    'endpoint:{conf_id}:{subkey}:{key_id}' or 'endpoint:{conf_id}:global:{key_id}'.")
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
        return None

    def _merge_default(self, p, dp):
        """Add skipped key from default.

        * Copy skipped sections from dp.
        * Copy skipped subkeys for existed in dp conf, zero position if
        multiple.
        """
        for section_id in dp:
            if section_id not in p:
                # Copy skipped section_ids from dp.
                p[section_id] = copy.deepcopy(dp[section_id])
            else:
                # Copy skipped subkeys for existed in dp conf(zero position).
                # Get zero position dp conf
                dp_conf_id = list(dp[section_id].keys())
                for conf_id, conf in p[section_id].items():
                    for subkey in dp[section_id][dp_conf_id].keys():
                        if subkey not in conf:
                            conf[subkey] = copy.deepcopy(
                                dp[section_id][dp_conf_id][subkey])
        return p

    # [future]
    def _find_new_keys(self, p, dp):
        """Find keys that not exist in dp."""
        new_keys = set()
        for key in list(p.keys()):
            if key not in dp:
                new_keys.add(key)
        if new_keys:
            # user can create configuration for arbitrary param
            # check if dict type
            for key in new_keys:
                if not isinstance(p[key], dict):
                    raise TypeError(f"Custom params[{key}]"
                                    f" should be the dict instance.")

    def _priority_arrange(self, res):
        """Sort configuration by `priority` sub-key."""
        min_heap = []
        for key in res:
            for subkey in res[key]:
                val = res[key][subkey]
                # TODO: [beta]
                # name = f'{key}__{subkey}'
                name = subkey
                priority = val.get('priority', 0)
                heapq.heappush(min_heap, (priority, (name, val)))
        sorted_ = heapq.nsmallest(len(min_heap), min_heap)
        return list(zip(*sorted_))[1]

    def _check_res(self, tup):
        """Check list of tuple for repeated values at first indices."""
        non_uniq = [k for (k, v) in
                    collections.Counter(list(zip(*tup))[0]).items() if v > 1]
        if non_uniq:
            raise ValueError(f"Non-unique configuration id found:\n"
                             f"    {non_uniq}")
        return None

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
