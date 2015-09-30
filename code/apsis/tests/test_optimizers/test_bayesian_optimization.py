__author__ = 'Frederik Diehl'

from apsis.optimizers.bayesian_optimization import BayesianOptimizer
from nose.tools import assert_is_none, assert_equal, assert_dict_equal, \
    assert_true, assert_false
from apsis.optimizers.bayesian.acquisition_functions import ExpectedImprovement, ProbabilityOfImprovement
from apsis.models.experiment import Experiment
from apsis.models.parameter_definition import MinMaxNumericParamDef, NominalParamDef
from apsis.models.candidate import Candidate
from apsis.utilities.import_utils import import_if_exists

class testBayesianOptimization(object):

    def test_init(self):
        #test default parameters
        exp = Experiment("test", {"x": MinMaxNumericParamDef(0, 1)})

        opt = BayesianOptimizer(exp)
        assert_equal(opt.initial_random_runs, 10)
        assert_is_none(opt.acquisition_hyperparams)
        assert_equal(opt.num_gp_restarts, 10)
        assert_true(isinstance(opt.acquisition_function, ExpectedImprovement))
        assert_dict_equal(opt.kernel_params, {})
        assert_equal(opt.kernel, "matern52")

        #test correct initialization
        opt_arguments = {
            "initial_random_runs": 5,
            "acquisition_hyperparams": {},
            "num_gp_restarts": 5,
            "acquisition": ProbabilityOfImprovement,
            "kernel_params": {},
            "kernel": "matern52",
            "mcmc": True,
        }
        opt = BayesianOptimizer(exp, opt_arguments)

        assert_equal(opt.initial_random_runs, 5)
        assert_dict_equal(opt.acquisition_hyperparams, {})
        assert_equal(opt.num_gp_restarts, 5)
        assert_true(isinstance(opt.acquisition_function, ProbabilityOfImprovement))
        assert_dict_equal(opt.kernel_params, {})
        assert_equal(opt.kernel, "matern52")

    def test_get_next_candidate(self):
        exp = Experiment("test", {"x": MinMaxNumericParamDef(0, 1),
                                  "y": NominalParamDef(["A", "B", "C"])})
        opt = BayesianOptimizer(exp, {"initial_random_runs": 3})
        for i in range(5):
            cand = opt.get_next_candidates()[0]
            assert_true(isinstance(cand, Candidate))
            cand.result = 2
            exp.add_finished(cand)
            opt.update(exp)
        cands = opt.get_next_candidates(num_candidates=3)
        assert_equal(len(cands), 3)