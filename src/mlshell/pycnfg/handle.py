"""The module contains class to read and execute configuration(s).

Configuration is a python dictionary.
It supports multiple sections.
Each section specify set of sub-configurations.
Each sub-configuration provide steps to construct an object, that can be
utilize as argument in some other sections.
Whole configuration could be passed to `pycnfg.run` or user-defined
wrapper around `pycnfg.Handler` to built underlying sub-configuration`s
objects one by one.

For each section there is common logic:
{'section_id':
    'configuration_id 1': {
        'init': initial object of custom type.
        'producer': factory class, which contain methods to run steps.
        'patch': add custom method to class.
        'steps': [
            ('method_id 1', {'kwarg_id':value, ..}),
            ('method_id 2', {'kwarg_id':value, ..}),
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
sections. To specify the order in which sections handled, the 'priority'
key is available in each sub-configuration

For flexibility, it is possible to:
* Specify global value for common kwargs in steps via `global` key.
* Create separate section for arbitrary parameter in steps.
* Monkey patch `producer` object with custom functions via `patch` key.
* `init` can be both an instance, a class or a function

Configuration keys
------------------
init : class or instance or function, optional (default={})
    Initial state for constructed object. Will be passed consecutive in steps
    as argument. If set as class or function `init()` will be auto called.

producer : class, optional (default=pycnfg.Producer)
    Factory to construct an object: `producer.produce(`init`,`steps`)`
    will be called, where `objects` is a dictionary with previously created
    objects {'section_id__configuration_id': object}. Class will be auto
    initialized with `producer(`objects`, 'section_id__configuration_id',
    **kwargs)`. If ('__init__', kwargs) step provided in `steps`, kwargs will
    be passed to initializer.

patch : dict {'method_id' : function}, optional (default={})
    Monkey-patching `producer` object with custom functions.

steps : list of tuples ('method_id', {**kwargs}), optional (default=[])
    List of class methods to run consecutive with kwargs.
    Each step should be a tuple: `('method_id', {kwargs to use})`,
    where 'method_id' should match to `producer` functions' names.
    It is possible to omit kwargs, in that case each step executed with kwargs
    set default in corresponding producer method (see `producer` interface).

    **kwargs : dict {'kwarg_id': value, ...}, optional (default={})
        Arguments depends on workflow methods.

        It is possible to create separate  section for any argument.
        Set `section_id__configuration_id` for kwarg value, then it would be
        auto-filled with corresponding section `objects` before step execution.
        To prevent auto substitution, use special '_id' postfix to `kwarg_id`.
        It is also possible to set list of `section_id__configuration_id`s.


        If `value` is set to None, parser try to resolve it. First it searches
        for value in `global`. Then resolver looks up 'kwarg_name'
        in section names. If such section exist, there are two possibilities:
        if `kwarg_name` contain '_id' postfix, resolver substitutes None with
        available `section_id__configuration_id`, otherwise with
        configuration object.
        If fails to find resolution, `value` is remained None. In case of
        resolution plurality, ValueError is raised.

priority : non-negative integer, optional (default=1)
    Priority of configuration execution. The more the higher priority.
    For two conf with same priority order is not guaranteed.
    If zero, not execute configuration.

global : dict {'kwarg_name': value, ...}, optional (default={})
    Specify values to resolve None for arbitrary kwargs. This is convenient for
    example when we use the same `pipeline` in all methods. It is not rewrite
    not-None values.

**keys : dict {'kwarg_name': value, ...}, optional (default={})
    All additional keys in configuration are moved to `global` automatically.
    If is useful if mostly rely on default configuration

Notes
-----
Default value can be reassigned in `pycnfg.Handler.read(conf, default)`.

Examples
--------
# Patch producer with custom functions.
def my_func(self, *args, **kwargs):
    # ... custom logic ...
    return res

{'patch': {'extra': my_func,},}

See Also
--------
:class:`pycnfg.Handler`:
    Reads configurations, executes steps.

"""


import collections
import copy
import heapq
import importlib.util
import inspect
import sys
import types

import mlshell.pycnfg as pycnfg

__all__ = ['Handler']


class Handler(object):
    """Read and execute configurations.

    Interface: read, exec.

    See Also
    ---------
    :class:`pycnfg.Producer`:
        Execute configuration steps.

    """
    _required_parameters = []

    def __init__(self):
        pass

    def read(self, conf, default_conf=None):
        """Read raw configuration and transform to executable.

        Parameters
        ----------
        conf : dict or str
            Set of configurations:
            {'section_id': {'configuration_id': configuration,},}.
            If str, absolute path to file with `CNFG` variable.
        default_conf : dict, optional (default=None)
            Set of default configurations:
            {'section_id': {'configuration_id': configuration, },}
            If None, read from `pycnfg.DEFAULT`.

        Returns
        -------
        configs : list of tuple [('section_id__configuration_id', config),...].
            List of configurations, prepared for execution.

        Notes
        -----
        Apply default:
        * If section is skipped, default section is used.
        * If sub-keys are skipped, default values are used for these sub-keys.

        Resolve kwargs:
        * If any step kwarg is None => use value from `global`,
        * if not in `global` => search `kwarg_id` in 'section_id's,
        If no section => remain None.
        If found section:
            If more then one configurations in section => ValueError.
            If `kwarg_id` contains postfix '_id', substitute None with
            `section_id__configuration_id`, otherwise with conf. object.

        """
        if isinstance(conf, str):
            spec = importlib.util.spec_from_file_location(f'CNFG', f"{conf}")
            conf_file = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(conf_file)
            # Otherwise problem with pickle, depends on the module path.
            sys.modules['CNFG'] = conf_file
            conf = copy.deepcopy(conf_file.conf)
        if default_conf is None:
            default_conf = copy.deepcopy(pycnfg.DEFAULT)
        configs = self._parse_conf(conf, default_conf)
        return configs

    def exec(self, configs):
        """Execute configurations in priority.

        For each configuration:
        * initialize producer
        `producer(`objects`, 'section_id__configuration_id', **kwargs)`,
        where kwargs took from ('__init__', kwargs) step if provided.
        * call `producer`.produce(`init`, `steps`).
        * store result under `section_id__configuration_id` in `objects`.

        Parameters
        ----------
        configs : list of tuple [('section_id__config__id', config), ...]
            List of configurations, prepared for execution.

        Returns
        -------
        objects : dict {'section_id__config__id', object,}
            Dictionary with resulted objects from `configs` execution.

        Notes
        -----
        `producer`/`init` auto initialized.

        Default values for skipped config keys:
        {'init': {}, 'steps': [], 'producer': pycnfg.Producer, 'patch': {},}

        """
        objects = {}
        for config in configs:
            oid, val = config
            objects[oid] = self._exec(oid, val, objects)
        return objects

    def _parse_conf(self, p, dp):
        # Apply default.
        p = self._merge_default(p, dp)
        # Resolve None inplace for configurations.
        # `ids` contain used confs ref by section.
        ids = {}  # {'section_id': set('configuration_id', )}
        for section_id in p:
            for conf_id in p[section_id]:
                self._resolve_none(p, section_id, conf_id, ids)

        # [deprecated] add -1 priority to skip conf.
        # in common config we need all objects
        # and explicit kwargs list not available.
        # # Remain only used configuration.
        # res = {}
        # for key, val in ids.items():
        #     if val:
        #         res[key] = {id_: p[key][id_] for id_ in val}
        # p = res

        # Arrange in priority.
        res = self._priority_arrange(p)  # [('section_id__config__id', config)]
        self._check_res(res)
        return res

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

    def _resolve_none(self, p, section_id, conf_id, ids):
        """Auto resolution for None parameters in endpoint section."""
        # [alternative] update with global when call step.
        # Assemble unknown keys to global.
        for key in p[section_id][conf_id]:
            if key not in ['init', 'producer', 'global',
                           'patch', 'steps', 'priority']:
                p[section_id][conf_id]['global']\
                    .update({key: p[section_id][conf_id].pop(key)})

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
                    self._substitute(p, section_id, conf_id, ids,
                                     key, glob_val, subkey, value, key_id)
        return None

    def _substitute(self, p, section_id, conf_id, ids,
                    key, glob_val, subkey, value, key_id):
        # If None use global.
        if not value[key_id]:
            # If global None use from conf (if only one provided)
            # for metrics not None is guaranteed before.
            if not glob_val:
                if len(p[key]) > 1:
                    raise ValueError(
                        f"Multiple {key} configurations provided,"
                        f" specify key:\n"
                        f"    '{section_id}:{conf_id}:{subkey}:{key_id}'"
                        f" or '{section_id}:{conf_id}:global:{key_id}'.")
                else:
                    glob_val = f"{key}__{list(p[key].keys())[0]}"
            value[key_id] = glob_val

        # Check if conf available.
        if isinstance(value[key_id], list):
            # for compatibility with sequence of ids (like metric)
            confs = [i.split('__')[-1] for i in value[key_id]]
        else:
            confs = [value[key_id].split('__')[-1]]
        for conf in confs:
            if conf not in p[key]:
                raise ValueError(f"Unknown configuration: {key}__{conf}.")
            elif p[key][conf]['priority'] == 0 \
                    and p[section_id][conf_id]['priority'] != 0:
                raise ValueError(f"Zero priority configuration {key}__{conf} "
                                 f"can`t be used in:\n"
                                 f"    {section_id}__{conf_id}__{subkey}.")

        # Substitute id(s).
        ids[key].update(confs)

        # [deprecated] Resolve in Producer.
        #   there are more reliable and consistent.
        # ACTUALLy ids can`t contains only names.
        # # Substitute either id(s), or conf.
        # if key_id.endswith('_id'):
        #     ids[key].update(confs)
        # else:
        #     # Set reference to `init`,
        #     # could be problem if object not ready.
        #     if len(confs) > 1:
        #         value[key_id] = [p[key][conf]['init'] for conf in confs]
        #     else:
        #         value[key_id] = p[key][confs[0]]['init']
        return None

    def _priority_arrange(self, res):
        """Sort configuration by `priority` sub-key."""
        min_heap = []
        for key in res:
            for subkey in res[key]:
                val = res[key][subkey]
                name = f'{key}__{subkey}'
                # [alternative]
                # name = subkey
                priority = val.get('priority', 1)
                if isinstance(priority, int) or priority < 0:
                    raise ValueError('Configuration priority should'
                                     ' be non-negative number.')
                if priority:
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

    def _exec(self, oid, conf, objects):
        init = conf.get('init', {})
        steps = conf.get('steps', [])
        producer = conf.get('producer', pycnfg.Producer)
        patch = conf.get('patch', {})
        if inspect.isclass(init):
            init = init()
        elif inspect.isfunction(init):
            init = init()
        if inspect.isclass(producer):
            kwargs = self._init_kwargs(steps)
            producer = producer(objects, oid, **kwargs)
        else:
            raise TypeError(f"{oid} producer should be a class.")
        producer = self._patch(patch, producer)
        return producer.produce(init, steps)

    def _init_kwargs(self, steps):
        try:
            kwargs = steps[0][1] if steps[0][0] == '__init__' else {}
        except IndexError:
            kwargs = {}
        return kwargs

    def _patch(self, patch, producer):
        """Monkey-patching producer.

        producer : class object
            Object to patch.
        patch : dict {'method_id' : function/existed 'method_id' }
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