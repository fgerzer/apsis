from sklearn.cross_validation import cross_val_score
from apsis.utilities.PreComputedGrid import PreComputedGrid
from apsis.evaluation.EvaluationFramework import EvaluationFramework
from apsis.models.ParamInformation import *
from apsis.RandomSearchCore import RandomSearchCore
from apsis.SimpleBayesianOptimizationCore import SimpleBayesianOptimizationCore
import logging
from apsis.adapters.SimpleScikitLearnAdapter import SimpleScikitLearnAdapter
import os
from sklearn import datasets
from sklearn.linear_model import ARDRegression
from sklearn.svm import SVR

def test_sklearn(filename, dim=100):
    param_defs = [
            LowerUpperNumericParamDef(0.1, 10)
        ]
    #logging.basicConfig(level=logging.DEBUG)
    if not os.path.isfile(filename):
        logging.info("Precomputing grid.")
        estimator = ARDRegression()
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        parameter_names = ["alpha"]
        obj_func_args = {"estimator": estimator,
                         "param_defs": param_defs,
                         "X": X,
                         "y": y,
                         "parameter_names": parameter_names}

        objective_function = objective_func_from_sklearn
        logging.info("Storing grid.")
        build_and_save_grid(filename, objective_function, param_defs, obj_func_args=obj_func_args, dimensionality=dim)

        logging.info("Grid stored.")
    grid = PreComputedGrid()
    grid.load_from_disk(filename)
    ev = EvaluationFramework()
    optimizers = [RandomSearchCore({"param_defs": param_defs}),
                  SimpleBayesianOptimizationCore({"param_defs": param_defs,
                                                  "initial_random_runs": 5})]
    steps = 25
    ev.evaluate_and_plot_precomputed_grid(optimizers, ["random", "bayes"], grid, steps, to_plot=["plot_evaluation_step_ranking", "best_result_per_step", "best_result_per_cost"])


def test_sklearn_alt(filename, dim=100):
    param_defs = [
            LowerUpperNumericParamDef(1e-10, 1e-2),
            LowerUpperNumericParamDef(1e-10, 1e-2),
            LowerUpperNumericParamDef(1e-10, 1e-2),
            LowerUpperNumericParamDef(1e-10, 1e-2)

        ]
    #logging.basicConfig(level=logging.DEBUG)
    if not os.path.isfile(filename):
        logging.info("Precomputing grid.")
        estimator = ARDRegression()
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        parameter_names = ["alpha_1", "alpha_2", "lambda_1", "lambda_2"]
        obj_func_args = {"estimator": estimator,
                         "param_defs": param_defs,
                         "X": X,
                         "y": y,
                         "parameter_names": parameter_names}

        objective_function = objective_func_from_sklearn
        logging.info("Storing grid.")
        build_and_save_grid(filename, objective_function, param_defs, obj_func_args=obj_func_args, dimensionality=dim)

        logging.info("Grid stored.")
    grid = PreComputedGrid()
    grid.load_from_disk(filename)
    ev = EvaluationFramework()
    optimizers = [RandomSearchCore({"param_defs": param_defs}),
                  SimpleBayesianOptimizationCore({"param_defs": param_defs,
                                                  "initial_random_runs": 5})]
    steps = 10
    ev.evaluate_and_plot_precomputed_grid(optimizers, ["random", "bayes"], grid, steps, to_plot=["plot_evaluation_step_ranking", "best_result_per_step", "best_result_per_cost"])


def test_sklearn_svm(filename, dim=100):
    param_defs = [
            LowerUpperNumericParamDef(0.01, 10),
            LowerUpperNumericParamDef(0, 1)

        ]
    #logging.basicConfig(level=logging.DEBUG)
    if not os.path.isfile(filename):
        logging.info("Precomputing grid.")
        estimator = SVR()
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        parameter_names = ["C", "epsilon"]
        obj_func_args = {"estimator": estimator,
                         "param_defs": param_defs,
                         "X": X,
                         "y": y,
                         "parameter_names": parameter_names}

        objective_function = objective_func_from_sklearn
        logging.info("Storing grid.")
        build_and_save_grid(filename, objective_function, param_defs, obj_func_args=obj_func_args, dimensionality=dim)

        logging.info("Grid stored.")
    grid = PreComputedGrid()
    grid.load_from_disk(filename)
    ev = EvaluationFramework()
    optimizers = [RandomSearchCore({"param_defs": param_defs}),
                  SimpleBayesianOptimizationCore({"param_defs": param_defs,
                                                  "initial_random_runs": 5})]
    steps = 20
    ev.evaluate_and_plot_precomputed_grid(optimizers, ["random", "bayes"], grid, steps, to_plot=["plot_evaluation_step_ranking", "best_result_per_step", "best_result_per_cost"])

def build_and_save_grid(filename, objective_function, param_defs, dimensionality=1000, obj_func_args=None):
    if obj_func_args is None:
        obj_func_args = {}
    grid = PreComputedGrid()
    grid.build_grid_points(param_defs=param_defs, dimensionality=dimensionality)
    grid.precompute_results(objective_function, obj_func_args)
    grid.save_to_disk(filename)

def objective_func_from_sklearn(candidate, estimator, param_defs, X, y, parameter_names, scoring="mean_squared_error", cv=3): #gets candidate, returns candidate.
    sk_ad = SimpleScikitLearnAdapter(estimator, param_defs, parameter_names=parameter_names)
    sk_learn_format = sk_ad.translate_vector_dict(candidate.params)
    estimator.set_params(**sk_learn_format)

    score = - cross_val_score(estimator, X, y, scoring=scoring, cv=cv).mean()
    #candidate.score = score
    return score


test_sklearn_svm("test.pickle", dim=100)