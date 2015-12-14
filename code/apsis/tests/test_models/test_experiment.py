__author__ = 'Frederik Diehl'

from apsis.models.experiment import Experiment
from nose.tools import assert_equal, assert_raises, assert_dict_equal, \
    assert_true, assert_false
from apsis.models.candidate import Candidate
from apsis.models.parameter_definition import *

class TestExperiment(object):
    exp = None

    def setup(self):
        name = "test_experiment"
        param_def = {
            "x": MinMaxNumericParamDef(0, 1),
            "name": NominalParamDef(["A", "B", "C"])
        }
        param_def_wrong = {
            "x": MinMaxNumericParamDef(0, 1),
            "name": ["A", "B", "C"]
        }
        minimization = True
        self.exp = Experiment(name, param_def, minimization)

        assert_equal(self.exp.name, name)
        assert_equal(self.exp.parameter_definitions, param_def)
        assert_equal(self.exp.minimization_problem, minimization)
        with assert_raises(ValueError):
            Experiment("fails", False)

        with assert_raises(ValueError):
            Experiment("fails too", param_def_wrong)


    def test_add(self):
        cand = Candidate({"x": 1, "name": "A"})

        cand_invalid = Candidate({"x": 1})
        cand_invalid2 = Candidate({"x": 2, "name": "A"})

        with assert_raises(ValueError):
            self.exp.add_pending(cand_invalid)
        with assert_raises(ValueError):
            self.exp.add_pending(cand_invalid2)


        self.exp.add_pending(cand)
        assert cand in self.exp.candidates_pending
        with assert_raises(ValueError):
            self.exp.add_pending(False)

        self.exp.add_finished(cand)
        assert cand in self.exp.candidates_finished
        with assert_raises(ValueError):
            self.exp.add_finished(False)

        cand2 = Candidate({"x": 0, "name": "B"})
        self.exp.add_working(cand2)
        assert cand2 in self.exp.candidates_working
        with assert_raises(ValueError):
            self.exp.add_working(False)

        self.exp.add_pausing(cand2)
        assert cand2 in self.exp.candidates_pending
        with assert_raises(ValueError):
            self.exp.add_pausing(False)

        self.exp.add_working(cand2)
        assert cand2 in self.exp.candidates_working
        with assert_raises(ValueError):
            self.exp.add_working(False)

        self.exp.add_finished(cand2)
        assert cand2 in self.exp.candidates_finished
        with assert_raises(ValueError):
            self.exp.add_finished(False)

    def test_better_cand(self):
        cand = Candidate({"x": 1, "name": "B"})
        cand2 = Candidate({"x": 0, "name": "A"})
        cand_none = Candidate({"x": 0.5, "name": "C"})
        cand_invalid = Candidate({"x": 0.5, "name": "D"})
        cand.result = 1
        cand2.result = 0
        assert_true(self.exp.better_cand(cand2, cand))
        assert_true(self.exp.better_cand(cand2, cand_none))
        self.exp.minimization_problem = False
        assert_true(self.exp.better_cand(cand, cand2))
        assert_false(self.exp.better_cand(cand2, cand))
        assert_true(self.exp.better_cand(cand, None))
        assert_false(self.exp.better_cand(None, cand))
        assert_false(self.exp.better_cand(None, None))
        with assert_raises(ValueError):
            self.exp.better_cand(cand, cand_invalid)
        with assert_raises(ValueError):
            self.exp.better_cand(cand_invalid, cand)
        with assert_raises(ValueError):
            self.exp.better_cand("fails", cand)
        with assert_raises(ValueError):
            self.exp.better_cand(cand, "fails")

    def test_warp(self):
        cand = Candidate({"x": 1})
        cand_out = self.exp.warp_pt_out(self.exp.warp_pt_in(cand.params))
        assert_dict_equal(cand.params, cand_out)

    def test_csv(self):
        cand = Candidate({"x": 1, "name": "A"})
        self.exp.add_finished(cand)
        string, steps_incl = self.exp.to_csv_results()
        assert_equal(steps_incl, 1)
        assert_equal(string, "step,id,name,x,cost,result,failed,best_result\n1,%s,A,1,None,None,False,None\n"%cand.id)

    def test_to_dict(self):
        cand = Candidate({"x": 1, "name": "A"})
        self.exp.add_finished(cand)
        self.exp.to_dict()

    def test_check_param_dict(self):
        param_dict = {"x": 1}
        assert_false(self.exp._check_param_dict(param_dict))

        param_dict = {"x": 1,
                      "name": "D"}
        assert_false(self.exp._check_param_dict(param_dict))

        param_dict = {"x": 1,
                      "name": "A"}
        assert_true(self.exp._check_param_dict(param_dict))