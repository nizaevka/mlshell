"""Start script example for ml workflow."""

import mlshell
import classes

# TODO: test two variants, one as API, one over data_id

# find project path/script name
project_path, script_name = mlshell.find_path()
# create logger
logger = mlshell.logger.CreateLogger(project_path, script_name).logger

# get params from conf.py
params = mlshell.GetParams(logger=logger).get_params(project_path)

# initialize Workflow class object
wf = mlshell.Workflow(project_path, logger=logger, params=params)

# list of data need to read, only if data_id workflow params
data_ids = set(params[key]['data_id'] for key in params['workflow'] if 'data_id' in params[key])

# TODO: not create data[key], only data['data_id'] from getparams predifined data_id list
data = {}
data_raw_columns = {}
data_base_plot = {}
setted = dict()  # {'data_id': 'fit'}
for key in ['fit', 'validate', 'gui', 'predict']:
    if params['workflow'][key]:
        data_id = params[key]['data_id']
        if data_id in setted:
            data[key] = data[setted['data_id']]
            continue
        if params['data'][data_id]['unify']:
            # will take from cache
            data[key] = None
        else:
            # get data from db (user-defined)
            data_raw = classes.GetData(logger=logger).get_data(**params['data'][data_id]['get'])
            # preprocess data for Workflow (user-defined)
            data[key], data_raw_columns[key], data_base_plot[key] = classes.DataPreprocessor(logger=logger).preprocess_data(**params['data'][data_id]['get'])
            # TODO: change preprocess to get categor features and targets
        # set data
        wf.set_data(data=data[key], data_id=params[key]['data_id'])  # self.data_df, self.categoric_ind_name, self.categoric_ind_name
        # TODO: need cache for all data, set here full dict, if any absent, use fit one
        #   but ram consumption?

        if 'split' is not False:
            # TODO: not split by default at all => 'split':Fasle, but {} mean True
            wf.split(data_id=data_id)
            # wf.split(data=data[key], **params['data'][data_id]['split'])
            # Internally self.data_df make dictionary!
        setted[data_id] = key

# TODO: for predict we need to set_data too, under apprpriate key from workflow => unify control

if params['workflow__load']:
    # load from disk
    wf.load(params['load']['file'])  # self.estimator
elif any(params['workflow'].values()):
    # create pipeline
    # TODO: API
    wf.create_pipeline()  # self.estimator

if params['workflow__fit']:
    # fit pipeline on train/tune hp if gs_flag=True
    wf.fit(**params['fit'], **params['gs'])

if params['dump']:
    # # dump on disk
    file = wf.dump()

if params['validate']:
    # validate predictions
    wf.validate()

if params['gui']:
    # generate param for gui module (only train data used)
    wf.gen_gui_params()

    # init gui object
    gui = mlshell.GUI(train_base_plot, wf.gui_params, logger=logger)

    # plot results
    gui.plot(base_sort=True)

if params['predict']:
    # read and preprocess test data
    test_raw = classes.GetData(logger=logger).get_data(*params['data__test__args'],
                                                       **params['data__test__kw_args'])
    test, test_raw_columns, test_base_plot = classes.DataPreprocessor(logger=logger).preprocess_data(test_raw)

    # make predictions on test data (auto unified)
    wf.predict(test, test_raw_columns)
