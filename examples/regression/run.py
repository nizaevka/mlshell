"""Start script example  for ml workflow"""

import mlshell
import classes

# find path
project_path, script_name = mlshell.find_path()
# create logger
logger = mlshell.logger.CreateLogger(f"{project_path}/logs_{script_name}").logger

# get params from conf.py
gp = mlshell.GetParams(project_path)

# get data from db (project specific)
gd = classes.GetData(logger=logger)
gd.get_data(filename=gp.params['train_file'], rows_limit=gp.params['rows_limit'], random_skip=gp.params['random_skip'])

# prepare data for analyse (project specific)
pp = classes.DataPreprocessor(gd.raw, logger=logger)

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

# validate prediction
wf.validate()

# dump on disk
file = wf.dump()

# load from disk
wf.load(file)

# read and preprocess new data
gd2 = classes.GetData(logger=logger)
gd2.get_data(filename=gp.params['test_file'], rows_limit=gp.params['rows_limit'])
pp2 = classes.DataPreprocessor(gd2.raw, logger=logger)

# make prediction to new data
wf.predict(pp2.data, pp2.raw_targets_names, pp2.raw_index_names)

# generate param for gui
wf.gen_gui_params()

# init gui object
gui = mlshell.GUI(pp.base_plot, wf.gui_params, logger=logger)

# plot results
gui.plot(isplot=True)
