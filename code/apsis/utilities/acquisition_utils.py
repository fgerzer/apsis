__author__ = 'Frederik Diehl'

from apsis.optimizers.bayesian import acquisition_functions
import numpy as np

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



def create_cand_matrix_vector(experiment, failed_treat):
    """
    Creates the candidate matrix and result vector.
    """
    parameter_warped_size = 0
    for p in experiment.parameter_definitions.values():
        parameter_warped_size += p.warped_size()

    if failed_treat[0] is "ignore":
        treated_candidates = 0
        for c in experiment.candidates_finished:
            if not c.failed:
                treated_candidates += 1
    else:
        treated_candidates = len(experiment.candidates_finished)

    candidate_matrix = np.zeros((treated_candidates,
                                 parameter_warped_size))
    results_vector = np.zeros((treated_candidates, 1))

    param_names = sorted(experiment.parameter_definitions.keys())

    best_candidate = None
    worst_candidate = None

    for c in experiment.candidates_finished:
        if experiment.better_cand(c, best_candidate):
            best_candidate = c
        if experiment.better_cand(worst_candidate, c):
            worst_candidate = c
    failed_value = 0
    if failed_treat[0] == "fixed_value":
        failed_value = failed_treat[1]
    elif failed_treat[1] == "worst_mult":
        failed_value = (worst_candidate.result - best_candidate.result) * \
                       failed_treat[1] + worst_candidate.result


    for i, c in enumerate(experiment.candidates_finished):
        warped_in = experiment.warp_pt_in(c.params)
        param_values = []
        for pn in param_names:
            param_values.extend(warped_in[pn])
        candidate_matrix[i, :] = param_values
        if c.failed:
            results_vector[i] = failed_value
        else:
            results_vector[i] = c.result
    return candidate_matrix, results_vector