__author__ = 'Frederik Diehl'

import logging
import os
from apsis.utilities.file_utils import ensure_directory_exists
import logging.config
import apsis

logging_intitialized = False

def get_logger(module, specific_log_name=None):
    """
    Abstraction from logging.getLogging, which also adds initialization.

    Logging is configured directly at root level (in the standard usecase, at
    least). You also have the opportunity to specify a certain directory to
    which details of only this logger (and all subloggers) are written.

    Currently, nothing is configurable from the outside. This is planned to be
    changed.

    Parameters
    ----------
    module : object or string
        The object for which we'd like to get the logger. The name of the
        logger is then, analogous to logging, set to
        module.__module__ + "." + module.__class__.__name__

        If the object is a string it will be taken as name directly.
    specific_log_name : string, optional
        If you want logging for this logger (and all sublogger) to a specific
        file, this allows you to set the corresponding filename.

    Returns
    -------
    logger: logging.logger
        A logging for module.
    """

    #if logger is already given as a string take directly. otherwise compute.
    if isinstance(module, basestring):
        new_logger_name = module
    else:
        new_logger_name = module.__module__ + "." + module.__class__.__name__

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    LOG_ROOT = os.environ.get('APSIS_LOG_ROOT', '/tmp/APSIS_WRITING/logs')
    ensure_directory_exists(LOG_ROOT)
    global logging_intitialized
    if not logging_intitialized:
        logging_intitialized = True
        #initialize the root logger.
        project_dirname = os.path.dirname(apsis.__file__)
        log_config_file = os.path.join(project_dirname, 'config/logging.conf')
        logging.config.fileConfig(log_config_file, defaults={'logfilename': os.path.join(LOG_ROOT, "log")})

    logger_existed = False
    if new_logger_name in logging.Logger.manager.loggerDict:
        logger_existed = True
    logger = logging.getLogger(new_logger_name)
    if specific_log_name is not None and not logger_existed:
        fh = logging.FileHandler(os.path.join(LOG_ROOT, specific_log_name))
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger