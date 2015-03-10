from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn.svm import NuSVC, SVC, libsvm
import os
import logging
from apsis.assistants.lab_assistant import PrettyLabAssistant, ValidationLabAssistant
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

from apsis.models.parameter_definition import *
from apsis.utilities.logging_utils import get_logger
from apsis.utilities.randomization import check_random_state

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


def evaluate_on_mnist(optimizer_names, optimizers, param_defs, optimizer_args, regressor, cv=10, percentage=1., steps=50, random_steps=10, plot_at_end=True, disable_auto_plot=False):
    """
    This evaluates the (pre-initialized) optimizers on a percentage of mnist.

    Parameters
    ----------
    LAss : LabAssistant
        The LabAssistant containing all of the experiments.
    optimizers : list of strings
        The optimizer names used. The first has to be the random optimizer.
    percentage : float, between 0 and 1, optional
        The percentage of MNIST on which we want to evaluate.
    """
    #TODO add args to comment.
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

    LAss = ValidationLabAssistant(cv=cv, disable_auto_plot=disable_auto_plot)

    #create random optimizer - assume this is the first one
    LAss.init_experiment(optimizer_names[0], optimizers[0], param_defs, minimization=False, optimizer_arguments=optimizer_args[0])

    #evaluate random search for 10 steps use these steps as init value for bayesian
    for i in range(random_steps * cv):
        print("random step " + str(i))
        do_evaluation(LAss, optimizer_names[0], regressor, mnist_data_train, mnist_data_test, mnist_target_train, mnist_target_test)

    #now clone experiment for each optimizer
    for j in range(1, len(optimizers)):
        LAss.clone_experiments_by_name(exp_name=optimizer_names[0], new_exp_name=optimizer_names[j], optimizer=optimizers[j], optimizer_arguments=optimizer_args[j])

    logger.info("Random Initialization Phase Finished.")
    logger.info("Competitive Evaluation Phase starts now.")

    #from there on go step by step all models
    for i in range(random_steps * cv, steps * cv):
        logger.info("Doing step %i" %i)
        for n in optimizer_names:
            print("normal step " + str(i) +  " for " + str(n))
            do_evaluation(LAss, n, regressor, mnist_data_train, mnist_data_test, mnist_target_train, mnist_target_test)

    #finally do an evaluation
    for n in optimizer_names:
        logger.info("Best %s score:  %s" %(n, LAss.get_best_candidate(n).result))

    if plot_at_end:
        LAss.plot_result_per_step(optimizer_names)
        LAss.plot_validation(optimizer_names)

def demo_MNIST(steps, random_steps, percentage, cv, plot_at_end=True, disable_auto_plot=False):
    logging.basicConfig(level=logging.DEBUG)
    regressor = SVC(kernel="poly")
    param_defs = {
        "C": MinMaxNumericParamDef(0,10),
        #"degree": FixedValueParamDef([1, 2, 3]),
        "gamma":MinMaxNumericParamDef(0, 1),
        "coef0": MinMaxNumericParamDef(0,1)
    }

    random_state_rs=check_random_state(42)

    optimizer_names = ["RandomSearch", "BayOpt_EI"]
    optmizers = ["RandomSearch", "BayOpt"]
    optimizer_args = [{"random_state": random_state_rs}, {"initial_random_runs": random_steps}]

    evaluate_on_mnist(optimizer_names, optmizers, param_defs, optimizer_args, regressor, cv, percentage, steps=steps, random_steps=random_steps,  plot_at_end=plot_at_end, disable_auto_plot=disable_auto_plot)

def demo_MNIST_LibSVM(steps, random_steps, percentage, cv, plot_at_end=True, disable_auto_plot=False):
    logging.basicConfig(level=logging.DEBUG)

    param_defs = {
        "C": MinMaxNumericParamDef(0,10),
        "gamma":MinMaxNumericParamDef(0, 1),
        "nu": MinMaxNumericParamDef(0,1)
    }
    random_state_rs=check_random_state(42)

    regressor = libsvm()

    optimizer_names = ["RandomSearch", "BayOpt_EI"]
    optmizers = ["RandomSearch", "BayOpt"]
    optimizer_args = [{"random_state": random_state_rs}, {"initial_random_runs": random_steps}]

    evaluate_on_mnist(optimizer_names, optmizers, param_defs, optimizer_args, regressor, cv, percentage, steps=steps, random_steps=random_steps,  plot_at_end=plot_at_end, disable_auto_plot=disable_auto_plot)


if __name__ == '__main__':
    demo_MNIST(20, 5, 0.01, 10, disable_auto_plot=True)