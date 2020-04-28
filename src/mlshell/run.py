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

    # init workflow with methods
    # take zero position
    endpoint_id = params['workflow']['endpoint_id']
    if not isinstance(endpoint_id, str):
        endpoint_id = endpoint_id[0]

    if params['endpoint'][endpoint_id]['class']:
        cls = params['endpoint'][endpoint_id]['class']
    else:
        cls = mlshell.Workflow

    wf = cls(project_path, logger=logger, params=params, data=None, pipeline=None)
    for key, val in params['endpoint'][endpoint_id].items():
        if isinstance(val, dict) and 'func' in val and val['func']:
            setattr(wf, key, types.MethodType(val['func'], wf))

    # TODO: actually read data only when call (ram consumption)
    ## read datasets
    data = {}
    for data_id in params['data']:
        if params['data'][data_id]['class']:
            cls = params['data'][data_id]['class']
        else:
            cls = mlshell.DataHandler
        handler = cls(project_path, logger=logger)
        for key, val in params['data'][data_id].items():
            if isinstance(val, dict) and 'func' in val and val['func']:
                setattr(handler, key, types.MethodType(val['func'], handler))
            data[data_id] = handler.handle(data_id, val)  # map(handler.handle, [val])

    ## set data
    wf.set_data(data)

    # [deprecated] load/create pipeline better made part of workflow steps
    ## read pipelines
    # PipelinHandler?
    ## set pipelines
    # wf.set_pipeline(data)

    #[deprecated] move to DataHandler class
    # wf.set_data(data_id=data_id, data=data)  # self.data_df, self.categoric_ind_name, self.categoric_ind_name
# TODO: need cache for all data, set here full dict, if any absent, use fit one
#   but ram consumption?

if 'split' is not False:
    # TODO: not split by default at all => 'split':Fasle, but {} mean True
    wf.split(data_id=data_id)
    # wf.split(data=data[key], **params['data'][data_id]['split'])

    # create/load pipelines
    mutable objects
    * internall data(pos_label and splits add to data in set_data, update TThrehold classifier correspondingly),
    * pipelines

    # call workflow methods
    func(self, **kwargs)



    TODO: pos_label
    pass


