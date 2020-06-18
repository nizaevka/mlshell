"""
The module includes utility to run preconfigured execution.
"""


import mlshell


def run(conf, default_conf=None, project_path=None, logger=None):
    """

    Parameters
    ----------
    conf : dict
        Configuration to pass in `mlshell.ConfHandler().read()`.
    default_conf : None, dict, optional (default=None)
        Default configurations to pass in `ConfHandler().read()`.
    project_path: str, optional (default='')
        Absolute path to current project dir.
        If None, auto detected by `mlshell.find_path()`.
    logger : None, logger object (default=None)
        If None, `mlshell.logger.CreateLogger()` will be used with script name.

    Returns
    -------
    objects : dict {'configuration_id': object}.
        Dict of objects created by execution all configurations.

    See Also
    --------
    :class:`mlshell.ConfHandler`:
        Reads configurations, executes steps.
    :class:`mlshell.CreateLogger`:
        Creates universal logger.
    :callback:`mlshell.find_path`:
        Finds start script directory and name.

    """
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
    return objects
