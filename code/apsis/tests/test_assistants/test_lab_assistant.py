__author__ = 'Frederik Diehl'

from apsis.assistants.lab_assistant import *
from nose.tools import assert_equal, assert_items_equal, assert_dict_equal, \
    assert_is_none, assert_raises, raises, assert_greater_equal, \
    assert_less_equal, assert_in
from apsis.utilities.logging_utils import get_logger
from apsis.models.parameter_definition import *

class TestLabAssistant(object):
    """
    Tests the lab_assistants.
    """

    def test_init(self):
        """
        Tests whether the initialization works correctly.
        Tests:
            - Whether the directory for writing is correct
            - exp_assistants is empty
            - logger name is correctly set.
        """
        LAss = PrettyLabAssistant()
        assert_equal(LAss.write_directory_base, "/tmp/APSIS_WRITING")
        assert_items_equal(LAss.exp_assistants, {})
        assert_equal(LAss.logger,
                     get_logger("apsis.assistants.lab_assistant.PrettyLabAssistant"))


    def test_init_experiment(self):
        """
        Tests whether the initialization works correctly.
        Tests:
            - optimizer correct
            - minimization correct
            - param_defs correct
            - No two experiments with the same name
        """
        optimizer = "RandomSearch"
        name = "test_init_experiment"
        param_defs = {
            "x": MinMaxNumericParamDef(0, 1),
            "name": NominalParamDef(["A", "B", "C"])
        }
        minimization = True

        LAss = PrettyLabAssistant()
        LAss.init_experiment(name, optimizer, param_defs, minimization=minimization)

        exp_ass = LAss.exp_assistants[name]

        assert_equal(exp_ass.optimizer, optimizer)
        assert_is_none(exp_ass.optimizer_arguments, None)
        assert_equal(exp_ass.experiment.minimization_problem, minimization)
        with assert_raises(ValueError):
            LAss.init_experiment(name, optimizer, param_defs, minimization=minimization)

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

        LAss = PrettyLabAssistant()
        LAss.init_experiment(name, optimizer, param_defs, minimization=minimization)
        cand = LAss.get_next_candidate(name)
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
        """
        optimizer = "RandomSearch"
        name = "test_init_experiment"
        param_defs = {
            "x": MinMaxNumericParamDef(0, 1),
            "name": NominalParamDef(["A", "B", "C"])
        }
        minimization = True

        LAss = PrettyLabAssistant()
        LAss.init_experiment(name, optimizer, param_defs, minimization=minimization)
        cand = LAss.get_next_candidate(name)
        cand.result = 1
        LAss.update(name, cand)
        assert_items_equal(LAss.exp_assistants[name].experiment.candidates_finished, [cand])
        assert_equal(LAss.exp_assistants[name].experiment.candidates_finished[0].result, 1)

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

        LAss = PrettyLabAssistant()
        LAss.init_experiment(name, optimizer, param_defs, minimization=minimization)
        cand_one = LAss.get_next_candidate(name)
        cand_one.result = 1
        LAss.update(name, cand_one)

        cand_two = LAss.get_next_candidate(name)
        cand_two.result = 0
        LAss.update(name, cand_two)

        assert_equal(cand_two, LAss.get_best_candidate(name))

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

        LAss = PrettyLabAssistant()
        LAss.init_experiment(name, optimizer, param_defs, minimization=minimization)
        LAss.init_experiment(name + "2", optimizer, param_defs, minimization=minimization)
        cand = LAss.get_next_candidate(name)
        cand.result = 1
        LAss.update(name, cand)
        LAss.write_out_plots_current_step()
        LAss.plot_result_per_step([name], show_plot=False)
        LAss.exp_assistants[name].experiment.minimization_problem = False
        LAss.plot_result_per_step(name, show_plot=False)

    def test_validation_lab_assistant(self):
        """
        Just a short test on whether validation lab assistant does not crash.
        """
        optimizer = "RandomSearch"
        name = "test_init_experiment"
        param_defs = {
            "x": MinMaxNumericParamDef(0, 1),
            "name": NominalParamDef(["A", "B", "C"])
        }
        minimization = True

        LAss = ValidationLabAssistant()
        LAss.init_experiment(name, optimizer, param_defs, minimization=minimization)
        LAss.init_experiment(name + "2", optimizer, param_defs, minimization=minimization)
        cand = LAss.get_next_candidate(name)
        cand.result = 1
        LAss.update(name, cand)
        LAss.write_out_plots_current_step()
        LAss.plot_result_per_step([name], show_plot=False)
        LAss.plot_validation([name], show_plot=False)
        LAss.exp_assistants[name][0].experiment.minimization_problem = False
        LAss.plot_result_per_step(name, show_plot=False)