__author__ = 'Frederik Diehl'

import logging
import os
from apsis.utilities.file_utils import ensure_directory_exists
import logging.config
import apsis
import yaml

logging_intitialized = False
testing = False


def get_logger(module, extra_info=None, save_path=None):
    """
    Abstraction from logging.getLogging, which also adds initialization.

    This loads the logging config from config/logging.conf.

    Parameters
    ----------
    module : object or string
        The object for which we'd like to get the logger. The name of the
        logger is then, analogous to logging, set to
        module.__module__ + "." + module.__class__.__name__
        If the object is a string it will be taken as name directly.

    extra_info : string, optional
        If None (the default), a usual logger is returned. If not, a
        logger_adapter is returned, which always prepends the corresponding
        string.

    save_path : string, optional
        The path on which to store the logging. If logging has been initialized
        previously, this is ignored (and a warning is logged). If a path has
        been specified in the config file, this is also ignored (and a warning
        is issued). Otherwise, this path replaces all instances of the token
        <SAVE_PATH> in the file_name of all handlers.
        If it does not end with "/" we'll automatically add it. That means both
        "/tmp/APSIS_WRITING" and "/tmp/APSIS_WRITING/" is treated identically,
        and logging is added in "/tmp/APSIS_WRITING/logs".

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

    global testing

    global logging_intitialized
    if not logging_intitialized and not testing:
        logging_intitialized = True

        # Look for the logging config file.
        project_dirname = os.path.dirname(apsis.__file__)
        log_config_file = os.path.join(project_dirname, 'config/logging.conf')
        with open(log_config_file, "r") as conf_file:
            conf_dict = yaml.load(conf_file)
        handlers = conf_dict["handlers"]
        handler_keys = handlers.keys()
        for h in handler_keys:
            if "filename" in handlers[h]:
                if "<SAVE_PATH>" in handlers[h]["filename"]:
                    if not save_path.endswith("/"):
                        save_path += "/"
                    handlers[h]["filename"] = handlers[h]["filename"].replace(
                        "<SAVE_PATH>/", save_path).replace("<SAVE_PATH>",
                                                           save_path)
                ensure_directory_exists(os.path.dirname(handlers[h]["filename"]))

        logging.config.dictConfig(conf_dict)

    logger = logging.getLogger(new_logger_name)

    if extra_info:
        logger = AddInfoClass(logger, {"extra_info": extra_info})

    return logger


def logging_tests():
    global testing
    print("Setting logging to testing.")
    testing = True


class AddInfoClass(logging.LoggerAdapter):
        def process(self, msg, kwargs):
            return '[%s] %s' % (self.extra['extra_info'], msg), kwargs