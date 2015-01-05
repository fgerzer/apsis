__author__ = 'Frederik Diehl'

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import cross_val_score
from apsis.assistants.experiment_assistant import PrettyExperimentAssistant
from apsis.models.parameter_definition import *
from apsis.assistants.lab_assistant import BasicLabAssistant, PrettyLabAssistant
import math
from sklearn.svm import NuSVC, SVC
import logging


def test_function():
    def func(x, y):
        #return x*math.cos(10*y)
        a=2
        return (a-x)**2 + 100*(y-x**2)**2

    param_defs = {
        "x": LowerUpperNumericParamDef(-100, 100),
        "y": LowerUpperNumericParamDef(-100, 100)
    }

    #logging.basicConfig(level=logging.DEBUG)

    LAss = PrettyLabAssistant()

    LAss.init_experiment("rand", "RandomSearch", param_defs, minimization=True)
    LAss.init_experiment("bay", "BayOpt", param_defs, minimization=True)

    results = []

    for i in range(100):
        to_eval = LAss.get_next_candidate("rand")
        print(to_eval)
        result = func(to_eval.params["x"], to_eval.params["y"])
        results.append(result)
        to_eval.result = result
        LAss.update("rand", to_eval)

        to_eval = LAss.get_next_candidate("bay")
        print(to_eval)
        result = func(to_eval.params["x"], to_eval.params["y"])
        results.append(result)
        to_eval.result = result
        LAss.update("bay", to_eval)

    print("Best bay score:  %s" %LAss.get_best_candidate("bay").result)
    print("Best bay:  %s" %LAss.get_best_candidate("bay"))
    print("Best rand score: %s" %LAss.get_best_candidate("rand").result)
    print("Best rand:  %s" %LAss.get_best_candidate("rand"))
    #x, y, z = BAss._best_result_per_step_data()
    LAss.plot_result_per_step(["rand", "bay"], plot_at_least=0.1)

def test_boston():
    boston_data = datasets.load_boston()
    regressor = SVC(kernel="poly")
    param_defs = {
        "C": LowerUpperNumericParamDef(0,10),
        "degree": FixedValueParamDef([1,2,3]),
        "gamma": LowerUpperNumericParamDef(0, 10),
        "coef0": LowerUpperNumericParamDef(0,10)
    }

    LAss = PrettyLabAssistant()

    LAss.init_experiment("rand", "RandomSearch", param_defs, minimization=False)
    LAss.init_experiment("bay", "BayOpt", param_defs, minimization=False)


    for i in range(20):
        to_eval = LAss.get_next_candidate("rand")
        print("rand" + str(to_eval.params))
        regressor.set_params(**to_eval.params)
        scores = cross_val_score(regressor, boston_data.data, boston_data.target,
                                 scoring="mean_squared_error", cv=3)
        result = scores.mean()
        to_eval.result = result
        LAss.update("rand", to_eval)

        to_eval = LAss.get_next_candidate("bay")
        print("bay" + str(to_eval.params))
        regressor.set_params(**to_eval.params)
        scores = cross_val_score(regressor, boston_data.data, boston_data.target,
                                 scoring="mean_squared_error", cv=3)
        result = scores.mean()
        to_eval.result = result
        LAss.update("bay", to_eval)

    print("================================")
    print("RAND")
    print(LAss.exp_assistants["rand"].experiment.candidates_finished)
    print("===================================")
    print("BAY")
    print(LAss.exp_assistants["bay"].experiment.candidates_finished)
    print("===================================")
    print("Best bay score:  %s" %LAss.get_best_candidate("bay").result)
    print("Best rand score: %s" %LAss.get_best_candidate("rand").result)
    #x, y, z = BAss._best_result_per_step_data()
    LAss.plot_result_per_step(["rand", "bay"], plot_at_least=1)

#test_boston()
test_function()