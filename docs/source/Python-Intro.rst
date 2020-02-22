Python intro
============

Create project dir “project_dir”, inside put:
•	“data” dir with raw datasheet
•	conf.py project`s current run params
•	params={
    'split_train_size': 0.7,
    'cv_splitter': model_selection.KFold(shuffle=False),
    'debug_pipeline': False,
    'isneed_cache':False,
    'cache_update': True,
    'isneeddump':False,
    'runs': None,
    'estimator_type': 'regressor',
    'main_estimator': linear_model.LinearRegression(),
    'hp_grid': {},
    'gs_verbose': 1,
    'th_strategy':0,
    'pos_label':1,
    'del_duplicates':False,
    'plot_analisys':False,
}
•	classes.py
GetData(logger) class
.get_data() method
load in memory raw data
.raw attribute
	result
Preprocessor(logger,raw_data) class
	.data attribute
		data after preprocessing
	.params attribute
		project`s current run params from conf.py




