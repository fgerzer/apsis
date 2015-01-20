__author__ = 'Andreas Jauch'

from apsis.models.parameter_definition import *
from apsis.assistants.lab_assistant import BasicLabAssistant, PrettyLabAssistant
import logging
from apsis.utilities.benchmark_functions import branin_func
from apsis.optimizers.bayesian.acquisition_functions import ExpectedImprovement
from apsis.models.candidate import Candidate
import nose.tools as nt
import scipy

class TestAcquisition(object):
    #LOG_FILENAME = "/tmp/APSIS_WRITING/logs/TestNumericParamDef.log"

    def test__translate_dict_vector(self):
        ei = ExpectedImprovement()

        param = {
            "y": 1,
            "x": 2
        }

        cand = Candidate(param)
        vec = ei._translate_dict_vector(cand.params)

        assert vec[0] == param['x'] and vec[1] == param['y']

    def test__translate_vector_dict(self):
        ei = ExpectedImprovement()

        names = ['y', 'x']
        vec = [2,1]

        param_dict = ei._translate_vector_dict(x_vector=vec,param_names=names)

        assert vec[0] == param_dict['x'] and vec[1] == param_dict['y']


    def test_expected_improvement_optimization(self):
        param_defs = {
            "x": LowerUpperNumericParamDef(-5, 10),
            "y": LowerUpperNumericParamDef(0, 15)
        }


        #logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger("test_acquisition")
        logger.info("Running test_acquisition")

        LAss = PrettyLabAssistant()

        LAss.init_experiment("rand", "RandomSearch", param_defs, minimization=True)
        LAss.init_experiment("bay_rand", "BayOpt", param_defs, minimization=True, optimizer_arguments={"initial_random_runs": 5, "mcmc": False, "acquisition_hyperparams":{"optimization": "random"}})
        LAss.init_experiment("bay_bfgs", "BayOpt", param_defs, minimization=True, optimizer_arguments={"initial_random_runs": 5, "mcmc": False, "acquisition_hyperparams":{"optimization": "bfgs"}} )

        results = []

        #evaluate all experiments one step at the time.
        for i in range(15):
            to_eval = LAss.get_next_candidate("rand")
            result = branin_func(to_eval.params["x"], to_eval.params["y"])
            results.append(result)
            to_eval.result = result
            print(to_eval)
            LAss.update("rand", to_eval)

            to_eval = LAss.get_next_candidate("bay_rand")
            result = branin_func(to_eval.params["x"], to_eval.params["y"])
            results.append(result)
            to_eval.result = result
            print(to_eval)
            LAss.update("bay_rand", to_eval)

            to_eval = LAss.get_next_candidate("bay_bfgs")
            result = branin_func(to_eval.params["x"], to_eval.params["y"])
            results.append(result)
            to_eval.result = result
            print(to_eval)
            LAss.update("bay_bfgs", to_eval)

        print("Best rand score: %s" %LAss.get_best_candidate("rand").result)
        print("Best rand:  %s" %LAss.get_best_candidate("rand"))
        print("Best bay_rand:  %s" %LAss.get_best_candidate("bay_rand").result)
        print("Best bay_rand:  %s" %LAss.get_best_candidate("bay_rand"))
        print("Best bay_bfgs score: %s" %LAss.get_best_candidate("bay_bfgs").result)
        print("Best bay_bfgs:  %s" %LAss.get_best_candidate("bay_bfgs"))
        #x, y, z = BAss._best_result_per_step_data()
        LAss.plot_result_per_step(["rand", "bay_rand", "bay_bfgs"], plot_at_least=1)

        real_min = 0.397887
        #check for real min: LAss.get_best_candidate("bay_rand").result

        return True


    # def test_ei_gradient(self):
    #TODO replace this by a manual check with scipy.optimize.approx_fprime
    #     param_defs = {
    #         "x": LowerUpperNumericParamDef(-5, 10),
    #         "y": LowerUpperNumericParamDef(0, 15)
    #     }
    #     LAss = PrettyLabAssistant()
    #     LAss.init_experiment("bay_rand", "BayOpt", param_defs, minimization=True, optimizer_arguments={"initial_random_runs": 2, "mcmc": False, "acquisition_hyperparams":{"optimization": "random"}})
    #     ei = ExpectedImprovement()
    #
    #     for i in range(50):
    #         to_eval = LAss.get_next_candidate("bay_rand")
    #         result = branin_func(to_eval.params["x"], to_eval.params["y"])
    #         to_eval.result = result
    #         print(to_eval)
    #         LAss.update("bay_rand", to_eval)
    #
    #         #now check gradient
    #         if LAss.exp_assistants['bay_rand'].optimizer.gp is not None:
    #             to_eval_vec = ei._translate_dict_vector(to_eval.params)
    #             grad_err = scipy.optimize.check_grad(ei._evaluate_vector, ei._evaluate_vector_gradient, to_eval_vec, tuple([LAss.exp_assistants['bay_rand'].optimizer.gp, LAss.exp_assistants['bay_rand'].experiment]))
    #             print("gradient err")
    #             print(grad_err)

