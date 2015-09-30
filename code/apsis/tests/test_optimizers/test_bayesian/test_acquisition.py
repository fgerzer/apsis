__author__ = 'Frederik Diehl'

from apsis.optimizers.bayesian_optimization import BayesianOptimizer
from nose.tools import assert_is_none, assert_equal, assert_dict_equal, \
    assert_true, assert_false
from apsis.optimizers.bayesian.acquisition_functions import ExpectedImprovement, ProbabilityOfImprovement
from apsis.models.experiment import Experiment
from apsis.models.parameter_definition import MinMaxNumericParamDef
from apsis.models.candidate import Candidate

class testAcquisitionFunction(object):

    def test_EI(self):
        exp = Experiment("test", {"x": MinMaxNumericParamDef(0, 1)})
        opt = BayesianOptimizer(exp, {"initial_random_runs": 3,
                                      "max_searcher": "LBFGSB"})

        for i in range(3):
            cands = opt.get_next_candidates(2)
            cand_one = cands[0]
            cand_two = cands[1]
            assert_true(isinstance(cand_one, Candidate))
            cand_one.result = 2
            exp.add_finished(cand_one)
            opt.update(exp)
            assert_true(isinstance(cand_two, Candidate))
            cand_two.result = 2
            exp.add_finished(cand_two)
            opt.update(exp)
        cands = opt.get_next_candidates(num_candidates=3)
        assert_equal(len(cands), 3)

    def test_PoI(self):
        exp = Experiment("test", {"x": MinMaxNumericParamDef(0, 1)})
        opt = BayesianOptimizer(exp, {"initial_random_runs": 3, "acquisition": ProbabilityOfImprovement})
        assert_true(isinstance(opt.acquisition_function, ProbabilityOfImprovement))

        for i in range(3):
            cands = opt.get_next_candidates(2)
            cand_one = cands[0]
            cand_two = cands[1]
            assert_true(isinstance(cand_one, Candidate))
            cand_one.result = 2
            exp.add_finished(cand_one)
            opt.update(exp)
            assert_true(isinstance(cand_two, Candidate))
            cand_two.result = 2
            exp.add_finished(cand_two)
            opt.update(exp)
        cands = opt.get_next_candidates(num_candidates=3)
        assert_equal(len(cands), 3)