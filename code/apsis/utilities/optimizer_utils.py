__author__ = 'Frederik Diehl'

#TODO imports

AVAILABLE_OPTIMIZERS = {}

def check_optimizer(optimizer, optimizer_arguments=None):
    """
    Returns the optimizer corresponding to optimizer.

    If optimizer is an Optimizer, no action is taken.
    If optimizer is a string corresponding to one of the optimizer names in
    AVAILABLE_OPTIMIZERS, the corresponding optimizer is initialized.
    Otherwise, raises a ValueError.

    Parameters
    ----------
    optimizer: Optimizer or String
        The optimizer - or string representing it - to return.
    optimizer_arguments: dict or None:
        The parameters for the optimizer. Will not be used if optimizer is
        already an optimizer

    Returns
    -------
    An Optimizer instance corresponding to optimizer.
    """
    if isinstance(optimizer, Optimizer):
        return optimizer

    if isinstance(optimizer, str) and optimizer in AVAILABLE_OPTIMIZERS.keys():
        return AVAILABLE_OPTIMIZERS[optimizer](optimizer_arguments)

    raise ValueError("No corresponding optimizer found for %s"
                     %str(optimizer))