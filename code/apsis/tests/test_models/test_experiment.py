__author__ = 'Frederik Diehl'

from apsis.models.experiment import Experiment
from nose.tools import assert_equal, assert_raises, assert_dict_equal, \
    assert_true, assert_false
from apsis.models.candidate import Candidate
from apsis.models.parameter_definition import *

class TestExperiment(object):

    def test_init(self):
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
        exp = Experiment(name, param_def, minimization)

        assert_equal(exp.name, name)
        assert_equal(exp.parameter_definitions, param_def)
        assert_equal(exp.minimization_problem, minimization)
        with assert_raises(ValueError):
            Experiment("fails", False)

        with assert_raises(ValueError):
            Experiment("fails too", param_def_wrong)

    def test_add(self):
        name = "test_experiment"
        param_def = {
            "x": MinMaxNumericParamDef(0, 1),
            "name": NominalParamDef(["A", "B", "C"])
        }
        minimization = True
        exp = Experiment(name, param_def, minimization)
        cand = Candidate({"x": 1, "name": "A"})

        cand_invalid = Candidate({"x": 1})
        cand_invalid2 = Candidate({"x": 2, "name": "A"})

        with assert_raises(ValueError):
            exp.add_pending(cand_invalid)
        with assert_raises(ValueError):
            exp.add_pending(cand_invalid2)


        exp.add_pending(cand)
        assert cand in exp.candidates_pending
        with assert_raises(ValueError):
            exp.add_pending(False)

        exp.add_finished(cand)
        assert cand in exp.candidates_finished
        with assert_raises(ValueError):
            exp.add_finished(False)

        cand2 = Candidate({"x": 0, "name": "B"})
        exp.add_working(cand2)
        assert cand2 in exp.candidates_working
        with assert_raises(ValueError):
            exp.add_working(False)

        exp.add_pausing(cand2)
        assert cand2 in exp.candidates_pending
        with assert_raises(ValueError):
            exp.add_pausing(False)

        exp.add_working(cand2)
        assert cand2 in exp.candidates_working
        with assert_raises(ValueError):
            exp.add_working(False)

        exp.add_finished(cand2)
        assert cand2 in exp.candidates_finished
        with assert_raises(ValueError):
            exp.add_finished(False)

    def test_better_cand(self):
        name = "test_experiment"
        param_def = {
            "x": MinMaxNumericParamDef(0, 1),
            "name": NominalParamDef(["A", "B", "C"])
        }
        minimization = True
        exp = Experiment(name, param_def, minimization)
        cand = Candidate({"x": 1, "name": "B"})
        cand2 = Candidate({"x": 0, "name": "A"})
        cand_none = Candidate({"x": 0.5, "name": "C"})
        cand.result = 1
        cand2.result = 0
        assert_true(exp.better_cand(cand2, cand))
        assert_true(exp.better_cand(cand2, cand_none))
        exp.minimization_problem = False
        assert_true(exp.better_cand(cand, cand2))
        assert_false(exp.better_cand(cand2, cand))

    def test_warp(self):
        name = "test_experiment"
        param_def = {
            "x": MinMaxNumericParamDef(0, 1),
            "name": NominalParamDef(["A", "B", "C"])
        }
        minimization = True
        exp = Experiment(name, param_def, minimization)
        cand = Candidate({"x": 1})
        cand_out = exp.warp_pt_out(exp.warp_pt_in(cand.params))
        assert_dict_equal(cand.params, cand_out)

    def test_csv(self):
        name = "test_experiment"
        param_def = {
            "x": MinMaxNumericParamDef(0, 1),
            "name": NominalParamDef(["A", "B", "C"])
        }
        minimization = True
        exp = Experiment(name, param_def, minimization)
        cand = Candidate({"x": 1, "name": "A"})
        exp.add_finished(cand)
        string, steps_incl = exp.to_csv_results()
        assert_equal(steps_incl, 1)
        assert_equal(string, "step,name,x,cost,result,best_result\n1,A,1,None,None,None\n")