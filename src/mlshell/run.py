""" """

import mlshell


def run(conf, default_conf=None, project_path='', logger=None):
    logger_name = 'logger'
    if not project_path:
        project_path, script_name = mlshell.find_path()
        logger_name = script_name
    if not logger:
        logger = mlshell.logger.CreateLogger(project_path, logger_name).logger

    # get params from conf.py
    handler = mlshell.ConfHandler(project_path=project_path, logger=logger)
    configs = handler.read(conf=conf, default_conf=default_conf)
    objects = handler.exec(configs)

# [deprecated] now unified.
#     # TODO: maybe hide inside classes monkeypatching or in read conf?
#     # prepare datasets.
#     datasets = {}
#     for data_id in params['dataset']:
#         if params['dataset'][data_id]['class']:
#             cls = params['dataset'][data_id]['class']
#         else:
#             cls = mlshell.DataFactory
#         factory = cls(project_path, logger=logger)
#         for key, val in params['dataset'][data_id].get('steps', {}):
#             if isinstance(val, dict) and 'func' in val and val['func']:
#                 setattr(factory, key, types.MethodType(val['func'], factory))
#         datasets[data_id] = factory.produce(data_id, params['dataset'][data_id])
#
#     # prepare pipelines.
#     pipelines = {}
#     for pipeline_id in params['pipeline']:
#         if params['pipeline'][pipeline_id]['class']:
#             cls = params['pipeline'][pipeline_id]['class']
#         else:
#             cls = mlshell.PipeFactory
#         factory = cls(project_path, logger=logger)
#         for key, val in params['pipeline'][pipeline_id].items():
#             if isinstance(val, dict) and 'func' in val and val['func']:
#                 setattr(factory, key, types.MethodType(val['func'], factory))
#         pipelines[pipeline_id] = factory.produce(pipeline_id, params['pipeline'][pipeline_id])
#
#     # prepare metrics.
#     metrics = params.get('metric', {})
#
#     # init workflow with methods
#     # endpoint take zero position
#     endpoint_id = params['workflow']['endpoint_id']
#     if not isinstance(endpoint_id, str):
#         endpoint_id = endpoint_id[0]
#     endpoint_params = params['endpoint'][endpoint_id]
#
#     if endpoint_params['global'].get('class', None):
#         cls = endpoint_params['global']['class']
#     else:
#         cls = mlshell.Workflow
#     wf = cls(project_path, logger=logger, endpoint_id=endpoint_id,
#              datasets=datasets, pipelines=pipelines, metrics=metrics, params=endpoint_params)
#
#     needs_resolve = []
#     # update/add new methods
#     for key, val in endpoint_params.items():
#         if isinstance(val, dict) and 'func' in val and val['func']:
#             if isinstance(val['func'], str):
#                 needs_resolve.append((key, val['func']))
#                 val['func'] = wf.__getattribute__(val['func'])
#             setattr(wf, key, types.MethodType(val['func'], wf))
#     # resolve str name for existed methods
#     for key, name in needs_resolve:
#         setattr(wf, key, getattr(wf, name))
#
#     # call steps
#     # TODO: looks the same as produce step for data, pipeline.
#     steps = params['workflow'].get('steps', {})
#     for step in steps:
#         kw_args = endpoint_params[step[0]]
#         new_kw_args = step[1] if len(step) > 1 else {}
#         kw_args.update(new_kw_args)
#         getattr(wf, step[0])(**kw_args)

