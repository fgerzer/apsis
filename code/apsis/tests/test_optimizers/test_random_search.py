__author__ = 'Frederik Diehl'

from apsis.optimizers.random_search import RandomSearch
from nose.tools import assert_is_none, assert_equal, assert_dict_equal, \
    assert_true, assert_false
from apsis.models.experiment import Experiment
from apsis.models.parameter_definition import MinMaxNumericParamDef, NominalParamDef
from apsis.models.candidate import Candidate


class test_RandomSearch(object):
    def test_init(self):
        #test initialization
        exp = Experiment("test", {"x": MinMaxNumericParamDef(0, 1)}, NominalParamDef(["A", "B", "C"]))
        opt = RandomSearch(exp)

    def test_get_next_candidate(self):
        exp = Experiment("test", {"x": MinMaxNumericParamDef(0, 1)}, NominalParamDef(["A", "B", "C"]))
        opt = RandomSearch(exp, {"initial_random_runs": 3})

        for i in range(5):
            cand = opt.get_next_candidates()[0]
            assert_true(isinstance(cand, Candidate))
            cand.result = 2
            exp.add_finished(cand)
        cands = opt.get_next_candidates(num_candidates=3)
        assert_equal(len(cands), 3)
