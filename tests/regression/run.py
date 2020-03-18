"""Example module for ml workflow"""

import mlshell

if 'GetData' not in globals() or 'DataPreprocessor' not in globals():
    from classes import GetData, DataPreprocessor

# find path
project_path, script_name = mlshell.find_path(__file__)
# create logger
logger = mlshell.logger.CreateLogger(project_path, script_name).logger

# get params from conf.py
gp = mlshell.GetParams(project_path)

# get data from db (project specific)
gd = GetData(logger=logger)
gd.get_data(*gp.params['get_data']['train']['args'], **gp.params['get_data']['train']['kw_args'])

# prepare data for analyse (project specific)
pp = DataPreprocessor(gd.raw, logger=logger)

# initialize object of Workflow class (encode/unify data included)
wf = mlshell.Workflow(project_path, pp.data, logger=logger, params=gp.params)

# analyse on whole data
# wf.before_split_analyze()

# create pipeline
wf.create_pipeline()  # self.estimator

# split data
wf.split()  # => self.train, self.test

# fit pipeline on train (tune hp if GS_flag=True)
wf.fit(gs_flag=gp.params['gs_flag'])

# test prediction
wf.validate()

# dump on disk
file = wf.dump()

# load from disk
wf.load(file)

# read and preprocess new data
gd2 = GetData(logger=logger)
gd2.get_data(*gp.params['get_data']['test']['args'], **gp.params['get_data']['test']['kw_args'])
pp2 = DataPreprocessor(gd2.raw, logger=logger)

# make prediction to new data
wf.predict(pp2.data, pp2.raw_targets_names, pp2.raw_index_names)

# # generate param for gui
# wf.gen_gui_params()
#
# # init gui object
# gui = GUI(pp.base_plot, wf.gui_params, logger=logger)
#
# gui.plot(isplot=True)
