__author__ = 'Frederik Diehl'

from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn.svm import NuSVC, SVC
import os
import logging
from apsis.assistants.lab_assistant import PrettyLabAssistant
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

from apsis.models.parameter_definition import *
from apsis.utilities.logging_utils import get_logger

logger = get_logger("demos.demo_MNIST")

# The goal of this demo is to run an optimization on MNIST with three
# different optimizers. It is used for comparing these optimizers.
# If you want to try different optimizers or regressors, change them in the
# lowest part of this file.

def do_evaluation(LAss, name, regressor, mnist_data_train, mnist_data_test, mnist_target_train, mnist_target_test):
    """
    This does a single evaluation of the regressor.

    It gets the next candidate to evaluate, sets the parameters for the
    regressor, then trains it and predicts on the test data. Afterwards, it
    updates LAss with the new result.

    Parameters
    ----------
    LAss : LabAssistant
        The LabAssistant to use.
    name : string
        The name of the experiment for this evaluation
    mnist_data_train, mnist_data_test, mnist_target_train, mnist_target_test :
        Data list for train and test set, each for data and target. Used for
        evaluation.

    """
    to_eval = LAss.get_next_candidate(name)
    regressor.set_params(**to_eval.params)

    regressor = regressor.fit(mnist_data_train, mnist_target_train)

    predict = regressor.predict(mnist_data_test)

    result = accuracy_score(mnist_target_test, predict)
    to_eval.result = result

    LAss.update(name, to_eval)


def evaluate_on_mnist(LAss, optimizers, regressor, percentage=1., steps=10, plot=True):
    """
    This evaluates the (pre-initialized) optimizers on a percentage of mnist.

    Parameters
    ----------
    LAss : LabAssistant
        The LabAssistant containing all of the experiments.
    optimizers : list of strings
        The optimizer names used.
    percentage : float, between 0 and 1, optional
        The percentage of MNIST on which we want to evaluate.
    """
    #We first use the sklearn function to get the MNIST dataset. If cached,
    #we use the cached variant.
    if percentage < 0 or percentage > 1:
        raise ValueError("Percentage has to be between 0 and 1, is %f"
                         %percentage)

    mnist = fetch_mldata('MNIST original',
                         data_home=os.environ.get('MNIST_DATA_CACHE', '~/.mnist-cache'))

    mnist.data, mnist.target = shuffle(mnist.data, mnist.target, random_state=0)

    mnist.data = mnist.data[:int(percentage*mnist.data.shape[0])]
    mnist.target = mnist.target[:int(percentage*mnist.target.shape[0])]

    #train test split
    mnist_data_train, mnist_data_test, mnist_target_train, mnist_target_test = \
        train_test_split(mnist.data, mnist.target, test_size=0.1, random_state=42)



    #We do this for steps steps.
    for i in range(steps):
        logger.info("Doing step %i" %i)
        for n in optimizers:
            do_evaluation(LAss, n, regressor, mnist_data_train, mnist_data_test, mnist_target_train, mnist_target_test)

    #finally do an evaluation
    for n in optimizers:
        logger.info("Best %s score:  %s" %(n, LAss.get_best_candidate(n).result))

    if plot:
        LAss.plot_result_per_step(optimizers)

def demo_MNIST(steps, plot=True):
    logging.basicConfig(level=logging.DEBUG)
    regressor = SVC(kernel="poly")
    param_defs = {
        "C": MinMaxNumericParamDef(0,10),
        "degree": FixedValueParamDef([1, 2, 3]),
        "gamma":MinMaxNumericParamDef(0, 1),
        "coef0": MinMaxNumericParamDef(0,1)
    }
    LAss = PrettyLabAssistant()

    LAss.init_experiment("random_mnist", "RandomSearch", param_defs, minimization=False)
    LAss.init_experiment("bay_mnist_ei_rand", "BayOpt", param_defs, minimization=False)
    LAss.init_experiment("bay_mnist_ei_bfgs", "BayOpt", param_defs,
                         minimization=False, optimizer_arguments=
        {"acquisition_hyperparams":{"optimization": "BFGS"}})
    optimizers = ["random_mnist", "bay_mnist_ei_rand", "bay_mnist_ei_bfgs"]
    evaluate_on_mnist(LAss, optimizers, regressor, 0.01, steps=steps, plot=plot)

if __name__ == '__main__':
    demo_MNIST(50)