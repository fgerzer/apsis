#TODO write plotting functions.

__author__ = 'Frederik Diehl'

from apsis.assistants.experiment_assistant import ExperimentAssistant
from apsis.utilities.file_utils import ensure_directory_exists
import time
import datetime
import os
from apsis.utilities.logging_utils import get_logger


ACTIONS_FOR_EXP = ["get_next_candidate", "update", "get_best_candidate",
                   "get_all_candidates"]


class LabAssistant():
    exp_assistants = None

    _write_directory_base = None
    _lab_run_directory = None
    _global_start_date = None
    _logger = None



    def __init__(self, write_directory_base=None):
        self._logger = get_logger(self)
        if write_directory_base is None:
            if os.name == "nt":
                write_directory_base = os.path.relpath("APSIS_WRITING")
            else:
                write_directory_base = "/tmp/APSIS_WRITING"
        self._logger.info("Initializing lab assistant.")
        self._logger.info("Writing results to %s" %write_directory_base)
        self._write_directory_base = write_directory_base
        self._global_start_date = time.time()
        self._init_directory_structure()
        self.exp_assistants = {}
        self._logger.info("lab assistant successfully initialized.")

    def init_experiment(self, name, optimizer, param_defs,
                        optimizer_arguments=None, minimization=True):
        if name in self.exp_assistants:
            raise ValueError("Already an experiment with name %s registered."
                             %name)


        exp_ass = ExperimentAssistant(name, optimizer,
                            param_defs, optimizer_arguments=optimizer_arguments,
                            minimization=minimization,
                            write_directory_base=self._lab_run_directory,
                            csv_write_frequency=1)
        self.exp_assistants[name] = exp_ass
        self._logger.info("Experiment initialized successfully.")

    def _init_directory_structure(self):
        """
        Method to create the directory structure if not exists
        for results and plots writing
        """
        if self._lab_run_directory is None:
            date_name = datetime.datetime.utcfromtimestamp(
                self._global_start_date).strftime("%Y-%m-%d_%H:%M:%S")

            self._lab_run_directory = os.path.join(self._write_directory_base,
                                                  date_name)

            ensure_directory_exists(self._lab_run_directory)

    def get_candidates(self, experiment_id):
        return self.exp_assistants[experiment_id].get_candidates()

    def get_next_candidate(self, experiment_id):
        return self.exp_assistants[experiment_id].get_next_candidate()

    def get_best_candidate(self, experiment_id):
        return self.exp_assistants[experiment_id].get_next_candidate()

    def update(self, experiment_id, status, candidate):
        return self.exp_assistants[experiment_id].update(status=status,
                                                         candidate=candidate)