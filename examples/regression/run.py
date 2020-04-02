"""Start script example for ml workflow."""

import mlshell
import classes

# find project path/script name
project_path, script_name = mlshell.find_path()
# create logger
logger = mlshell.logger.CreateLogger(project_path, script_name).logger

# get params from conf.py
gp = mlshell.GetParams(project_path)

# get data from db (user-defined)
gd = classes.GetData(logger=logger)
gd.get_data(*gp.params['get_data']['train']['args'], **gp.params['get_data']['train']['kw_args'])

# prepare data for analyse (user-defined)
pp = classes.DataPreprocessor(gd.raw, logger=logger)

# initialize Workflow class object
wf = mlshell.Workflow(project_path, logger=logger, params=gp.params)
# unify data
wf.unify_data(data=pp.data)  # self.data_df

# create pipeline
wf.create_pipeline()  # self.estimator

# split data
wf.split()  # self.x_train, self.y_train, self.x_test, self.y_test

# fit pipeline on train/tune hp if gs_flag=True
wf.fit(gs_flag=gp.params['gs_flag'])

# validate predictions
wf.validate()

# dump on disk
file = wf.dump()

# load from disk
wf.load(file)

# read and preprocess new data
gd2 = classes.GetData(logger)
gd2.get_data(*gp.params['get_data']['test']['args'], **gp.params['get_data']['test']['kw_args'])
pp2 = classes.DataPreprocessor(gd2.raw, logger)

# make predictions on new data
wf.predict(pp2.data, pp2.raw_targets_names, pp2.raw_index_names)

# generate param for gui module
wf.gen_gui_params()

# init gui object
gui = mlshell.GUI(pp.base_plot, wf.gui_params, logger=logger)

# plot results
gui.plot(base_sort=True)
