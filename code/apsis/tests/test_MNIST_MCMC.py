__author__ = 'Frederik Diehl'

from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.svm import NuSVC, SVC
import os
import logging
import time
from apsis.assistants.lab_assistant import PrettyLabAssistant
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

from apsis.models.parameter_definition import *
from sklearn.linear_model import LassoLars
from sklearn.tree import DecisionTreeClassifier


def do_evaluation(LAss, name, regressor, mnist_data_train, mnist_data_test, mnist_target_train, mnist_target_test):
    to_eval = LAss.get_next_candidate(name)
    regressor.set_params(**to_eval.params)

    regressor = regressor.fit(mnist_data_train, mnist_target_train)

    predict = regressor.predict(mnist_data_test)

    result = accuracy_score(mnist_target_test, predict)
    to_eval.result  = result
    print(name + " " + str(result))
    print(name + " " + str(regressor.get_params()))
    print(name + " " + str(to_eval.params))
    print("======================================")
    LAss.update(name, to_eval)


def evaluate_on_mnist_mcmc_vs_likelyhood(percentage=1.):
    #load mnist dataset
    mnist = fetch_mldata('MNIST original',
                         data_home=os.environ.get('MNIST_DATA_CACHE', '~/.mnist-cache'))

    print("Mnist Data Size " + str(mnist.data.shape))
    print("Mnist Labels Size" + str(mnist.target.shape))
    mnist.data, mnist.target = shuffle(mnist.data, mnist.target, random_state=0)

    mnist.data = mnist.data[:int(percentage*mnist.data.shape[0])]
    mnist.target = mnist.target[:int(percentage*mnist.target.shape[0])]

    #train test split
    mnist_data_train, mnist_data_test, mnist_target_train, mnist_target_test = \
        train_test_split(mnist.data, mnist.target, test_size=0.1, random_state=42)

    print("Mnist Data Size " + str(mnist.data.shape))
    print("Mnist Labels Size" + str(mnist.target.shape))

    regressor = SVC(kernel="poly")
    param_defs = {
        "C": LowerUpperNumericParamDef(0,10),
        "degree": FixedValueParamDef([1, 2, 3]),
        "gamma":LowerUpperNumericParamDef(0, 1),
        "coef0": LowerUpperNumericParamDef(0,1)
    }

    #regressor = LassoLars()
    #regressor = DecisionTreeClassifier()

    #param_defs = {
    #    "max_features": LowerUpperNumericParamDef(0, 1),
    #    "max_depth": FixedValueParamDef([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    #    "min_samples_split": FixedValueParamDef([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    #    "min_samples_leaf": FixedValueParamDef([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    #}

    LAss = PrettyLabAssistant()

    LAss.init_experiment("random_mnist", "RandomSearch", param_defs, minimization=False)
    LAss.init_experiment("bay_mnist", "BayOpt", param_defs, minimization=False)
    LAss.init_experiment("bay_mcmc_mnist", "BayOpt", param_defs,
                         minimization=False, optimizer_arguments={"mcmc": True})

    steps = 50

    for i in range(steps):
        for n in ["random_mnist", "bay_mnist", "bay_mcmc_mnist"]:
            do_evaluation(LAss, n, regressor, mnist_data_train, mnist_data_test, mnist_target_train, mnist_target_test)

    #finally do an evaluation
    for n in ["random_mnist", "bay_mnist", "bay_mcmc_mnist"]:
        print("Best %s score:  %s" %(n, LAss.get_best_candidate(n).result))
    LAss.plot_result_per_step(["random_mnist", "bay_mnist", "bay_mcmc_mnist"], plot_at_least=1)


if __name__ == '__main__':
    evaluate_on_mnist_mcmc_vs_likelyhood(0.01)