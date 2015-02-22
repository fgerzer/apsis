__author__ = 'Frederik Diehl'

from apsis.optimizers.bayesian_optimization import SimpleBayesianOptimizer
from nose.tools import assert_is_none, assert_equal, assert_dict_equal, \
    assert_true, assert_false
from apsis.optimizers.bayesian.acquisition_functions import ExpectedImprovement, ProbabilityOfImprovement
from apsis.models.experiment import Experiment
from apsis.models.parameter_definition import MinMaxNumericParamDef
from apsis.models.candidate import Candidate

class testAcqusitionFunction(object):

    def test_EI(self):
        opt = SimpleBayesianOptimizer({"initial_random_runs": 3})
        exp = Experiment("test", {"x": MinMaxNumericParamDef(0, 1)})
        for i in range(10):
            cand = opt.get_next_candidates(exp)[0]
            assert_true(isinstance(cand, Candidate))
            cand.result = 2
            exp.add_finished(cand)
        cands = opt.get_next_candidates(exp, num_candidates=3)
        assert_equal(len(cands), 3)

    def test_PoI(self):
        opt = SimpleBayesianOptimizer({"initial_random_runs": 3, "acquisition": ProbabilityOfImprovement})
        assert_true(isinstance(opt.acquisition_function, ProbabilityOfImprovement))
        exp = Experiment("test", {"x": MinMaxNumericParamDef(0, 1)})
        for i in range(10):
            cand = opt.get_next_candidates(exp)[0]
            assert_true(isinstance(cand, Candidate))
            cand.result = 2
            exp.add_finished(cand)
        cands = opt.get_next_candidates(exp, num_candidates=3)
        assert_equal(len(cands), 3)