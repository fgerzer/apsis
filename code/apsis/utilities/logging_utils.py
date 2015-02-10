__author__ = 'Frederik Diehl'

import logging
import os
from apsis.utilities.file_utils import ensure_directory_exists

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
    module : object
        The object for which we'd like to get the logger. The name of the
        logger is then, analogous to logging, set to
        module.__module__ + "." + module.__class__.__name__
    specific_log_name : string, optional
        If you want logging for this logger (and all sublogger) to a specific
        file, this allows you to set the corresponding filename.

    Returns
    -------
    logger: logging.logger
        A logging for module.
    """

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if not logging_intitialized:
        #initialize the root logger.
        root_logger = logging.getLogger()
        LOG_ROOT = os.environ.get('APSIS_LOG_ROOT', '/tmp/APSIS_WRITING/logs')
        ensure_directory_exists(LOG_ROOT)
        fh_root = logging.FileHandler(os.path.join(LOG_ROOT, "log"))
        fh_root.setFormatter(formatter)
        fh_root.setLevel(logging.INFO)
        root_logger.addHandler(fh_root)
    else:
        LOG_ROOT = os.environ.get('APSIS_LOG_ROOT', '/tmp/APSIS_WRITING/logs')

    logger = logging.getLogger(module.__module__ + "." + module.__class__.__name__)
    if specific_log_name is not None:

        fh = logging.FileHandler(os.path.join(LOG_ROOT, specific_log_name))
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger