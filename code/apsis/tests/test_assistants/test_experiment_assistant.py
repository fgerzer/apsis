__author__ = 'Frederik Diehl'

from apsis.assistants.experiment_assistant import ExperimentAssistant
from nose.tools import assert_equal, assert_items_equal, assert_dict_equal, \
    assert_is_none, assert_raises, raises, assert_greater_equal, \
    assert_less_equal, assert_in, assert_true, assert_false, with_setup
from apsis.utilities.logging_utils import get_logger
from apsis.models.parameter_definition import *
from apsis.optimizers.random_search import QueueRandomSearch
import time

class TestExperimentAssistant(object):
    """
    Tests the experiment assistant.
    """
    EAss = None
    param_defs = None

    def setup(self):
        """
        Tests whether the initialization works correctly.
        Tests:
            - optimizer correct
            - minimization correct
            - param_defs correct
        """
        optimizer = "RandomSearch"
        name = "test_init_experiment"
        self.param_defs = {
            "x": MinMaxNumericParamDef(0, 1),
            "name": NominalParamDef(["A", "B", "C"])
        }
        minimization = True

        self.EAss = ExperimentAssistant(name, optimizer, self.param_defs, minimization=minimization)

        assert_equal(self.EAss.optimizer, optimizer)
        assert_is_none(self.EAss.optimizer_arguments, None)
        assert_equal(self.EAss.experiment.minimization_problem, minimization)


    def teardown(self):
        self.EAss.exit()

    def test_get_next_candidate(self):
        """
        Tests the get next candidate function.
        Tests:
            - The candidate's parameters are acceptable
        """

        cand = None
        counter = 0
        while cand is None and counter < 20:
            cand = self.EAss.get_next_candidate()
            time.sleep(0.1)
            counter += 1
        if counter == 20:
            raise Exception("Received no result in the first 2 seconds.")
        assert_is_none(cand.result)
        params = cand.params
        assert_less_equal(params["x"], 1)
        assert_greater_equal(params["x"], 0)
        assert_in(params["name"], self.param_defs["name"].values)

    def test_update(self):
        """
        Tests whether update works.
            - candidate exists in the list
            - result is equal
            - the status message incorrect error works
            - the candidate instance check works
        """
        cand = self.EAss.get_next_candidate()
        cand.result = 1
        self.EAss.update(cand)
        assert_items_equal(self.EAss.experiment.candidates_finished, [cand])
        assert_equal(self.EAss.experiment.candidates_finished[0].result, 1)

        self.EAss.update(cand, "pausing")
        self.EAss.update(cand, "working")
        with assert_raises(ValueError):
            self.EAss.update(cand, status="No status.")

        with assert_raises(ValueError):
            self.EAss.update(False)

    def test_get_best_candidate(self):
        """
        Tests whether get_best_candidate works.
            - Whether the best of the two candidates is the one it should be.
        """
        cand_one = self.EAss.get_next_candidate()
        cand_one.result = 1
        self.EAss.update(cand_one)

        cand_two = self.EAss.get_next_candidate()
        cand_two.result = 0
        self.EAss.update(cand_two)

        assert_equal(cand_two, self.EAss.get_best_candidate())

    def test_all_plots_working(self):
        """
        Tests whether all of the plot functions work. Does not test for correctness.
        """
        cand = self.EAss.get_next_candidate()
        cand.result = 1
        self.EAss.update(cand)

        cand = self.EAss.get_next_candidate()
        cand.result = 0

        cand = self.EAss.get_next_candidate()
        cand.result = 2
        self.EAss.plot_result_per_step(show_plot=False)