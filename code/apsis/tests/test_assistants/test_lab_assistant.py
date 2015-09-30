__author__ = 'Frederik Diehl'

from apsis.assistants.lab_assistant import *
from nose.tools import assert_equal, assert_items_equal, assert_dict_equal, \
    assert_is_none, assert_raises, raises, assert_greater_equal, \
    assert_less_equal, assert_in
from apsis.utilities.logging_utils import get_logger
from apsis.models.parameter_definition import *
import matplotlib.pyplot as plt

class TestLabAssistant(object):
    """
    Tests the lab_assistants.
    """

    LAss = None
    param_defs = None

    def setup(self):
        self.LAss = LabAssistant()

    def teardown(self):
        self.LAss.set_exit()

    def test_init(self):
        """
        Tests whether the initialization works correctly.
        Tests:
            - Whether the directory for writing is correct
            - _exp_assistants is empty
            - logger name is correctly set.
        """
        if os.name == "nt":
            assert_equal(self.LAss._write_directory_base, "/tmp/APSIS_WRITING")
        assert_items_equal(self.LAss.exp_assistants, {})
        assert_equal(self.LAss._logger,
                     get_logger("apsis.assistants.lab_assistant.LabAssistant"))


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
        self.param_defs = {
            "x": MinMaxNumericParamDef(0, 1),
            "name": NominalParamDef(["A", "B", "C"])
        }
        optimizer_arguments = {
            "multiprocessing": "none"
        }
        minimization = True

        exp_id = self.LAss.init_experiment(name, optimizer,
                                  optimizer_arguments=optimizer_arguments,
                                  param_defs=self.param_defs, minimization=minimization)

        exp_ass = self.LAss.exp_assistants[exp_id]

        assert_equal(exp_ass._optimizer.__class__.__name__, optimizer)
        assert_equal(exp_ass._optimizer_arguments, optimizer_arguments)
        assert_equal(exp_ass._experiment.minimization_problem, minimization)
        with assert_raises(ValueError):
            self.LAss.init_experiment(name, optimizer, exp_id=exp_id,
                              optimizer_arguments=optimizer_arguments,
                              param_defs=self.param_defs, minimization=minimization)
        return exp_id



    def test_get_next_candidate(self):
        """
        Tests the get next candidate function.
        Tests:
            - The candidate's parameters are acceptable
        """
        exp_id = self.test_init_experiment()
        cand = self.LAss.get_next_candidate(exp_id)
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
        """
        exp_id = self.test_init_experiment()
        cand = self.LAss.get_next_candidate(exp_id)
        cand.result = 1
        self.LAss.update(exp_id, status="finished", candidate=cand)
        assert_items_equal(self.LAss.exp_assistants[exp_id]._experiment.candidates_finished, [cand])
        assert_equal(self.LAss.exp_assistants[exp_id]._experiment.candidates_finished[0].result, 1)

    def test_get_best_candidate(self):
        """
        Tests whether get_best_candidate works.
            - Whether the best of the two candidates is the one it should be.
        """
        exp_id = self.test_init_experiment()
        cand_one = self.LAss.get_next_candidate(exp_id)
        cand_one.result = 1
        self.LAss.update(exp_id, "finished", cand_one)

        cand_two = self.LAss.get_next_candidate(exp_id)
        cand_two.result = 0
        self.LAss.update(exp_id, "finished", cand_two)

        assert_equal(cand_two, self.LAss.get_best_candidate(exp_id))

    def test_all_plots_working(self):
        """
        Tests whether all of the plot functions work. Does not test for correctness.
        """
        exp_id_one = self.test_init_experiment()
        exp_id_two = self.test_init_experiment()
        cand = self.LAss.get_next_candidate(exp_id_one)
        cand.result = 1
        self.LAss.update(exp_id_one, "finished", cand)
        self.LAss.write_out_plots_current_step()
        self.LAss.plot_result_per_step([exp_id_one])
        self.LAss.exp_assistants[exp_id_one]._experiment.minimization_problem = False
        self.LAss.plot_result_per_step(exp_id_two)


    def test_compute_current_step_overall(self):
        exp_id = self.test_init_experiment()
        self.LAss._compute_current_step_overall()


    def test_validation_lab_assistant(self):
        """
        Just a short test on whether validation lab assistant does not crash.
        """
        optimizer = "RandomSearch"

        name = "test_init_experiment"
        self.param_defs = {
            "x": MinMaxNumericParamDef(0, 1),
            "name": NominalParamDef(["A", "B", "C"])
        }
        optimizer_arguments = {
            "multiprocessing": "none"
        }
        minimization = True

        LAss = ValidationLabAssistant(cv=1)
        exp_id_one = LAss.init_experiment(name, optimizer, optimizer_arguments = optimizer_arguments,
                                          param_defs=self.param_defs, minimization=minimization)
        exp_id_two = LAss.init_experiment(name + "2", optimizer, optimizer_arguments = optimizer_arguments,
                                          param_defs=self.param_defs, minimization=minimization)
        cand = LAss.get_next_candidate(exp_id_one)
        cand.result = 1
        LAss.update(exp_id_one, "finished", cand)

        #cand = LAss.get_next_candidate(exp_id_two)
        #cand.result = 1
        #LAss.update(exp_id_two, "finished", cand)


        LAss.write_out_plots_current_step()
        LAss.plot_result_per_step([exp_id_one], show_plot=False)
        LAss.plot_validation([exp_id_one], show_plot=False)
        LAss.exp_assistants[exp_id_one][0]._experiment.minimization_problem = False
        LAss.plot_result_per_step(exp_id_one, show_plot=False)
        LAss._compute_current_step_overall()

        LAss.clone_experiments_by_id(exp_id_one, optimizer, optimizer_arguments, "new_name")
        LAss.set_exit()
