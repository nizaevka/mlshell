import mlshell


def run(params):
    # find project path/script name
    project_path, script_name = mlshell.find_path()
    # create logger
    logger = mlshell.logger.CreateLogger(project_path, script_name).logger

    # get params from conf.py
    params = mlshell.GetParams(logger=logger).get_params(project_path, params=params)

    pass