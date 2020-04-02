"""Example module for ml workflow"""

import mlshell
if 'GetData' not in globals() or 'DataPreprocessor' not in globals():
    from classes import GetData, DataPreprocessor


# find project path/script name
project_path, script_name = mlshell.find_path()
# create logger
logger = mlshell.logger.CreateLogger(project_path, script_name).logger

# get params from conf.py
params = mlshell.GetParams(logger=logger).get_params(project_path)

# get data from db (user-defined)
train_raw = GetData(logger=logger).get_data(*params['get_data']['train']['args'],
                                            **params['get_data']['train']['kw_args'])
# preprocess data for Workflow (user-defined)
train, train_raw_columns, train_base_plot = DataPreprocessor(logger=logger).preprocess_data(train_raw)

# initialize Workflow class object
wf = mlshell.Workflow(project_path, logger=logger, params=params)
# unify data
wf.unify_data(data=train)  # self.data_df

# create pipeline
wf.create_pipeline()  # self.estimator

# split data
wf.split()  # self.x_train, self.y_train, self.x_test, self.y_test

# fit pipeline on train/tune hp if gs_flag=True
wf.fit(gs_flag=params['gs_flag'])

# validate predictions
wf.validate()

# dump on disk
file = wf.dump()

# load from disk
wf.load(file)

# read and preprocess test data
test_raw = GetData(logger).get_data(*params['get_data']['test']['args'],
                                    **params['get_data']['test']['kw_args'])
test, test_raw_columns, test_base_plot = DataPreprocessor(logger=logger).preprocess_data(test_raw)

# make predictions on test data (auto unified)
wf.predict(test, test_raw_columns)

# generate param for gui module (only train data used)
wf.gen_gui_params()

# init gui object
gui = mlshell.GUI(train_base_plot, wf.gui_params, logger=logger)

# plot results
gui.plot(plot_flag=False, base_sort=True)
