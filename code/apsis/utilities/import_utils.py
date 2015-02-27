from apsis.utilities.logging_utils import get_logger

logger = get_logger("apsis.utils.import_utils")

def import_if_exists(module_name):
    """
    Function tries to import a module but will not fail if the module does
    not exist.

    Parameters
    ----------
    module_name : String
     The name of the module to be imported.

    Returns
    --------
    success : True
        Whether the module was successfully imported.
    module : module or None
        Returns the imported module iff successful, otherwise returns None.
    """
    try:
        module = __import__(module_name)
    except ImportError:
        logger.warning("Module " + str(module_name) +
                        " could not be imported as it could not be found.")
        return False, None
    else:
        return True, module