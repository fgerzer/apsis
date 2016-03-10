__author__ = 'Frederik Diehl'

from apsis.optimizers.bayesian import acquisition_functions

AVAILABLE_ACQUISITIONS = {
    "ExpectedImprovement": acquisition_functions.ExpectedImprovement,
    "ProbabilityOfImprovement": acquisition_functions.ProbabilityOfImprovement
}


def check_acquisition(acquisition, acquisition_params):
    """
    Checks whether optimizer is an acquisition function or builds one.

    Parameters
    ----------
    acquisition : string, AcquisitionFunction instance or class
        The acquisition function to initialize. If instance, the other
        parameters will be ignored.
    acquisition_params: dict, optional
        The parameters governing the behaviour of the acquisition function. If
        None, default values are used.

    Returns
    -------
    acquisition : AcquisitionFunction instance
        An initialized optimizer instance.

    Raises
    ------
    ValueError
        If the optimizer is a string, and one cannot find it in
        AVAILABLE_OPTIMIZERS. If not an optimizer subclass. If the
        multiprocessing argument is not an acceptable value.

    """
    if acquisition_params is None:
        acquisition_params = {}

    if isinstance(acquisition, acquisition_functions.AcquisitionFunction):
        return acquisition

    if isinstance(acquisition, basestring):
        try:
            acquisition = AVAILABLE_ACQUISITIONS[acquisition]
        except:
            raise ValueError("No corresponding acquisition found for %s. "
                             "Acquisition must be in %s" %(
                str(acquisition), AVAILABLE_ACQUISITIONS.keys()))

    if not issubclass(acquisition, acquisition_functions.AcquisitionFunction):
        raise ValueError("%s is of type %s, not AcquisitionFunction type."
                         %(acquisition, type(acquisition)))
    return acquisition(acquisition_params)
