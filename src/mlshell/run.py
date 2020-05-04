"""
TODO:
    move get, preprocess, plot to Workflow
    change preprocess to get categor features and targets

"""

import mlshell
import types


def run(params):
    # find project path/script name
    project_path, script_name = mlshell.find_path()
    # create logger
    logger = mlshell.logger.CreateLogger(project_path, script_name).logger

    # get params from conf.py
    params = mlshell.GetParams(logger=logger).get_params(project_path, params=params)

    # TODO: maybe hide inside classes monkeypatching or in read conf?
    # prepare datasets
    datasets = {}
    for data_id in params['data']:
        if params['data'][data_id]['class']:
            cls = params['data'][data_id]['class']
        else:
            cls = mlshell.DataFactory
        factory = cls(project_path, logger=logger)
        for key, val in params['data'][data_id].items():
            if isinstance(val, dict) and 'func' in val and val['func']:
                setattr(factory, key, types.MethodType(val['func'], factory))
        datasets[data_id] = factory.produce(data_id, params['data'][data_id])

    # prepare pipelines
    pipelines = {}
    for pipeline_id in params['pipeline']:
        if params['pipeline'][pipeline_id]['class']:
            cls = params['pipeline'][pipeline_id]['class']
        else:
            cls = mlshell.PipeFactory
        factory = cls(project_path, logger=logger)
        for key, val in params['pipeline'][pipeline_id].items():
            if isinstance(val, dict) and 'func' in val and val['func']:
                setattr(factory, key, types.MethodType(val['func'], factory))
        pipelines[pipeline_id] = factory.produce(pipeline_id, params['pipeline'][pipeline_id])

    # init workflow with methods
    # endpoint take zero position
    endpoint_id = params['workflow']['endpoint_id']
    if not isinstance(endpoint_id, str):
        endpoint_id = endpoint_id[0]
    if params['endpoint'][endpoint_id]['global']['class']:
        cls = params['endpoint'][endpoint_id]['global']['class']
    else:
        cls = mlshell.Workflow
    wf = cls(project_path, logger=logger, params=params, datasets=datasets, pipelines=pipelines)
    # TODO: move inside readconf, endpoint should be applied to class there, before creation
    for key, val in params['endpoint'][endpoint_id].items():
        if isinstance(val, dict) and 'func' in val and val['func']:
            setattr(wf, key, types.MethodType(val['func'], wf))

    # call steps

    # additional methods to change after creation
    # ## set data
    # wf.add_data(data)
    # wf.pop_data(data_id)
    # ## set pipelines
    # wf.add_pipeline(data)
    # wf.pop_pipeline(pipeline_id)
