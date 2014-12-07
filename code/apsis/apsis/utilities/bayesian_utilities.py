from apsis.bayesian.AcquisitionFunctions import *

acquisition_functions = {
    'ExpectedImprovement': ExpectedImprovement,
    'ProbabilityOfImprovement': ProbabilityOfImprovement
}

def check_acquisition(acq):
    if isinstance(acq, AcquisitionFunction):
        return acq
    if isinstance(acq, str) and acq in acquisition_functions.keys():
        return acquisition_functions[acq]

    raise ValueError("check_acquisition got %s as acq, which was neither a "
                     "valid AcquisitionFunction nor a valid string. Valid "
                     "strings are: %s" %(str(acq), str(acquisition_functions)))