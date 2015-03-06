__author__ = 'Frederik Diehl'

from sklearn.svm import NuSVC, SVC
from apsis.assistants.lab_assistant import PrettyLabAssistant, ValidationLabAssistant

from apsis.models.parameter_definition import *
from apsis.utilities.logging_utils import get_logger
import logging

from apsis.utilities.randomization import check_random_state
from demo_MNIST import evaluate_on_mnist

# The goal of this demo is to run an optimization on MNIST with several
# different optimizers. It is used for comparing these optimizers.
# If you want to try different optimizers or regressors, change them in the
# lowest part of this file.
# This is in a different file compared to demo_MNIST because MCMC requires
# quite a bit more time compared to non-MCMC.



def demo_MNIST_MCMC(steps, random_steps, percentage, cv, plot_at_end=True, disable_auto_plot=False):
    logging.basicConfig(level=logging.DEBUG)
    regressor = SVC(kernel="poly")
    param_defs = {
        "C": MinMaxNumericParamDef(0,10),
        #"degree": FixedValueParamDef([1, 2, 3]),
        "gamma":MinMaxNumericParamDef(0, 1),
        "coef0": MinMaxNumericParamDef(0,1)
    }

    random_state_rs= check_random_state(42)

    optimizer_names = ["RandomSearch", "BayOpt_EI", "BayOpt_EI_MCMC"]
    optmizers = ["RandomSearch", "BayOpt", "BayOpt"]
    optimizer_args = [{"random_state": random_state_rs}, {"initial_random_runs": random_steps}, {"initial_random_runs": random_steps, "mcmc": True}]

    evaluate_on_mnist( optimizer_names, optmizers, param_defs, optimizer_args, regressor, cv, percentage, steps=steps*cv, random_steps=random_steps,  plot_at_end=True, disable_auto_plot=disable_auto_plot)

if __name__ == '__main__':
    demo_MNIST_MCMC(20, 5, 0.01, 1, True, True)