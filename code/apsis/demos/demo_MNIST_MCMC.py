__author__ = 'Frederik Diehl'

from sklearn.svm import NuSVC, SVC
from apsis.assistants.lab_assistant import PrettyLabAssistant

from apsis.models.parameter_definition import *
from apsis.utilities.logging_utils import get_logger

from demo_MNIST import evaluate_on_mnist

# The goal of this demo is to run an optimization on MNIST with several
# different optimizers. It is used for comparing these optimizers.
# If you want to try different optimizers or regressors, change them in the
# lowest part of this file.
# This is in a different file compared to demo_MNIST because MCMC requires
# quite a bit more time compared to non-MCMC.

if __name__ == '__main__':


    LAss = PrettyLabAssistant()

    regressor = SVC(kernel="poly")
    param_defs = {
        "C": MinMaxNumericParamDef(0,10),
        "degree": FixedValueParamDef([1, 2, 3]),
        "gamma":MinMaxNumericParamDef(0, 1),
        "coef0": MinMaxNumericParamDef(0,1)
    }
    steps = 10
    LAss.init_experiment("random_mnist", "RandomSearch", param_defs, minimization=False)
    LAss.init_experiment("bay_mnist", "BayOpt", param_defs, minimization=False)
    LAss.init_experiment("bay_mcmc_mnist", "BayOpt", param_defs,
                         minimization=False, optimizer_arguments={"mcmc": True})
    optimizers = ["random_mnist", "bay_mnist", "bay_mcmc_mnist"]
    evaluate_on_mnist(LAss, optimizers, regressor, 0.01, steps=steps)