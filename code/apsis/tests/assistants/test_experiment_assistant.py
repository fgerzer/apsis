__author__ = 'Frederik Diehl'

from apsis.assistants.experiment_assistant import *
from nose.tools import assert_equal, assert_items_equal, assert_dict_equal, \
    assert_is_none, assert_raises, raises, assert_greater_equal, \
    assert_less_equal, assert_in
from apsis.utilities.logging_utils import get_logger
from apsis.models.parameter_definition import *

class TestAcquisition(object):
    """
    Tests the lab_assistants.
    """


    def test_init_experiment(self):
        """
        Tests whether the initialization works correctly.
        Tests:
            - optimizer correct
            - minimization correct
            - param_defs correct
        """
        optimizer = "RandomSearch"
        name = "test_init_experiment"
        param_defs = {
            "x": MinMaxNumericParamDef(0, 1),
            "name": NominalParamDef(["A", "B", "C"])
        }
        minimization = True

        EAss = PrettyExperimentAssistant(name, optimizer, param_defs, minimization=minimization)

        assert_equal(EAss.optimizer, optimizer)
        assert_is_none(EAss.optimizer_arguments, None)
        assert_equal(EAss.experiment.minimization_problem, minimization)

        EAss2 = PrettyExperimentAssistant(name, optimizer, param_defs, minimization=minimization, experiment_directory_base="/tmp/APSIS_WRITING")


    def test_get_next_candidate(self):
        """
        Tests the get next candidate function.
        Tests:
            - The candidate's parameters are acceptable
        """
        optimizer = "RandomSearch"
        name = "test_init_experiment"
        param_defs = {
            "x": MinMaxNumericParamDef(0, 1),
            "name": NominalParamDef(["A", "B", "C"])
        }
        minimization = True

        EAss = PrettyExperimentAssistant(name, optimizer, param_defs, minimization=minimization)
        cand = EAss.get_next_candidate()
        assert_is_none(cand.result)
        params = cand.params
        assert_less_equal(params["x"], 1)
        assert_greater_equal(params["x"], 0)
        assert_in(params["name"], param_defs["name"].values)

    def test_update(self):
        """
        Tests whether update works.
            - candidate exists in the list
            - result is equal
            - the status message incorrect error works
            - the candidate instance check works
        """
        optimizer = "RandomSearch"
        name = "test_init_experiment"
        param_defs = {
            "x": MinMaxNumericParamDef(0, 1),
            "name": NominalParamDef(["A", "B", "C"])
        }
        minimization = True

        EAss = PrettyExperimentAssistant(name, optimizer, param_defs, minimization=minimization)
        cand = EAss.get_next_candidate()
        cand.result = 1
        EAss.update(cand)
        assert_items_equal(EAss.experiment.candidates_finished, [cand])
        assert_equal(EAss.experiment.candidates_finished[0].result, 1)

        EAss.update(cand, "pausing")
        EAss.update(cand, "working")
        with assert_raises(ValueError):
            EAss.update(cand, status="No status.")

        with assert_raises(ValueError):
            EAss.update(False)

    def test_get_best_candidate(self):
        """
        Tests whether get_best_candidate works.
            - Whether the best of the two candidates is the one it should be.
        """
        optimizer = "RandomSearch"
        name = "test_init_experiment"
        param_defs = {
            "x": MinMaxNumericParamDef(0, 1),
            "name": NominalParamDef(["A", "B", "C"])
        }
        minimization = True

        EAss = PrettyExperimentAssistant(name, optimizer, param_defs, minimization=minimization)
        cand_one = EAss.get_next_candidate()
        cand_one.result = 1
        EAss.update(cand_one)

        cand_two = EAss.get_next_candidate()
        cand_two.result = 0
        EAss.update(cand_two)

        assert_equal(cand_two, EAss.get_best_candidate())

    def test_all_plots_working(self):
        """
        Tests whether all of the plot functions work. Does not test for correctness.
        """
        optimizer = "RandomSearch"
        name = "test_init_experiment"
        param_defs = {
            "x": MinMaxNumericParamDef(0, 1),
            "name": NominalParamDef(["A", "B", "C"])
        }
        minimization = True

        EAss = PrettyExperimentAssistant(name, optimizer, param_defs, minimization=minimization)
        cand = EAss.get_next_candidate()
        cand.result = 1
        EAss.update(cand)

        cand = EAss.get_next_candidate()
        cand.result = 0

        cand = EAss.get_next_candidate()
        cand.result = 2

        EAss.plot_result_per_step(show_plot=False)