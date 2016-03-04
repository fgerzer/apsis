__author__ = 'Frederik Diehl'

from apsis.assistants.experiment_assistant import ExperimentAssistant
from apsis.utilities.file_utils import ensure_directory_exists
import time
import datetime
import os
import json
from apsis.utilities.logging_utils import get_logger
from apsis.utilities.plot_utils import plot_lists, write_plot_to_file
import matplotlib.pyplot as plt
import uuid
import copy
import numpy as np
import apsis.models.experiment as experiment

# These are the colours supported by the plot.
COLORS = ["g", "r", "c", "b", "m", "y"]


class LabAssistant(object):
    """
    This is used to control multiple experiments at once.

    This is done by abstracting a dict of named experiment assistants.

    Attributes
    ----------
    _exp_assistants : dict of ExperimentAssistants.
        The dictionary of experiment assistants this LabAssistant uses.
    _write_dir : String, optional
        The directory to write all the results and plots to.
    _logger : logging.logger
        The logger for this class.
    """
    _exp_assistants = None

    _write_dir = None

    _global_start_date = None
    _logger = None

    def __init__(self, write_dir=None):
        """
        Initializes the lab assistant.

        Parameters
        ----------
        write_dir: string, optional
            Sets the write directory for the lab assistant. If None (default),
            nothing will be written.
        """
        self._logger = get_logger(self)
        self._logger.info("Initializing lab assistant.")
        self._logger.info("Writing results to %s" %write_dir)
        self._write_dir = write_dir

        self._exp_assistants = {}

        reloading_possible = True
        try:
            if self._write_dir:
                with open(self._write_dir + "/lab_assistant.json", "r"):
                    pass
            else:
                reloading_possible = False
        except IOError:
            reloading_possible = False

        if not reloading_possible:
            self._global_start_date = time.time()
        else:
            # set the correct path.
            with open(self._write_dir + "/lab_assistant.json", 'r') as infile:
                lab_assistant_json = json.load(infile)
            self._global_start_date = lab_assistant_json["global_start_date"]
            for p in lab_assistant_json["exp_assistants"].values():
                self._load_exp_assistant_from_path(p)

        self._write_state_to_file()
        self._logger.info("lab assistant successfully initialized.")

    def init_experiment(self, name, optimizer, param_defs, exp_id=None,
                        notes=None, optimizer_arguments=None,
                        minimization=True):
        """
        Initializes an experiment.

        Parameters
        ----------
        name : string
            name of the experiment.
        optimizer : string
            String representation of the optimizer.
        param_defs : dict of parameter definitions
            Dictionary of parameter definition classes.
        optimizer_arguments : dict, optional
            A dictionary defining the operation of the optimizer. See the
            respective documentation of the optimizers.
            Default is None, which are default values.
        exp_id : string or None, optional
            The id of the experiment, which will be used to reference it.
            Should be a proper uuid, and especially has to be unique. If it is
            not, an error may be returned.
        notes : jsonable object or None, optional
            Any note that you'd like to put in the experiment. Could be used
            to provide some details on the experiment, on the start time or the
            user starting it.
        minimization : bool, optional
            Whether the problem is one of minimization. Defaults to True.

        Returns
        -------
        exp_id : string
            String representing the id of the experiment or "failed" if failed.

        Raises
        ------
        ValueError :
            Iff there already is an experiment with the exp_id for this lab
            assistant. Does not occur if no exp_id is given.
        """
        if exp_id in self._exp_assistants.keys():
            raise ValueError("Already an experiment with id %s registered."
                             %exp_id)

        if exp_id is None:
            while True:
                exp_id = uuid.uuid4().hex
                if exp_id not in self._exp_assistants.keys():
                    break

        if not self._write_dir:
            exp_assistant_write_directory = None
        else:
            exp_assistant_write_directory = os.path.join(self._write_dir +
                                                     "/" + exp_id)
            ensure_directory_exists(exp_assistant_write_directory)

        exp = experiment.Experiment(name,
                                    param_defs,
                                    exp_id,
                                    notes,
                                    minimization)

        exp_ass = ExperimentAssistant(optimizer,
                                      experiment=exp,
                                      optimizer_arguments=optimizer_arguments,
                                      write_dir=exp_assistant_write_directory)
        self._exp_assistants[exp_id] = exp_ass
        self._logger.info("Experiment initialized successfully.")
        self._write_state_to_file()
        return exp_id

    def _load_exp_assistant_from_path(self, path):
        """
        This loads a complete exp_assistant from path.

        Specifically, it looks for exp_assistant.json in the path and restores
        optimizer_class, optimizer_arguments and write_dir from this. It then
        loads the experiment from the write_dir/experiment.json, then
        initializes both.

        Parameters
        ----------
        path : string
            The path from which to initialize. This must contain an
            exp_assistant.json as specified.
        """
        with open(path + "/exp_assistant.json", 'r') as infile:
            exp_assistant_json = json.load(infile)

        optimizer_class = exp_assistant_json["optimizer_class"]
        optimizer_arguments = exp_assistant_json["optimizer_arguments"]
        exp_ass_write_dir = exp_assistant_json["write_dir"]
        ensure_directory_exists(exp_ass_write_dir)
        exp = self._load_experiment(path)

        exp_ass = ExperimentAssistant(optimizer_class=optimizer_class,
                                      experiment=exp,
                                      optimizer_arguments=optimizer_arguments,
                                      write_dir=exp_ass_write_dir)

        if exp_ass.exp_id in self._exp_assistants:
            raise ValueError("Loaded exp_id is duplicated in experiment! id "
                             "is %s" %exp_ass.exp_id)
        self._exp_assistants[exp_ass.exp_id] = exp_ass

    def _load_experiment(self, path):
        """
        Loads an experiment from path.

        Looks for experiment.json in path.

        Parameters
        ----------
        path : string
            The path where experiment.json is located.
        """
        with open(path + "/experiment.json", 'r') as infile:
            exp_json = json.load(infile)
        exp = experiment.from_dict(exp_json)
        return exp


    def _write_state_to_file(self):
        """
        Writes the state of this lab assistant to a file.

        Iff _write_dir is not None, it will collate global_start_date and a
        dictionary of every experiment assistant, and dump this to
        self._write_dir/lab_assistant.json.
        """
        if not self._write_dir:
            return
        state = {"global_start_date": self._global_start_date,
                "exp_assistants": {x.exp_id: x.write_dir for x
                                    in self._exp_assistants.values()}}
        with open(self._write_dir + '/lab_assistant.json', 'w') as outfile:
            json.dump(state, outfile)

    def get_candidates(self, experiment_id):
        """
        Returns all candidates for a specific experiment.

        Parameters
        ----------
        experiment_id : string
            The id of the experiment for which to return the candidates.

        Returns
        -------
        result : dict
            A dictionary of three lists with the keys finished, pending and
            working, with the corresponding candidates.
        """
        return self._exp_assistants[experiment_id].get_candidates()

    def get_next_candidate(self, experiment_id):
        """
        Returns the next candidates for a specific experiment.

        Parameters
        ----------
        experiment_id : string
            The id of the experiment for which to return the next candidate.

        Returns
        -------
        next_candidate : Candidate or None
            The Candidate object that should be evaluated next. May be None,
            which is equivalent to no candidate generated.
        """
        return self._exp_assistants[experiment_id].get_next_candidate()

    def get_best_candidate(self, experiment_id):
        """
        Returns the best candidates for a specific experiment.

        Parameters
        ----------
        experiment_id : string
            The id of the experiment for which to return the best candidate.

        Returns
        -------
        next_candidate : Candidate or None
            The Candidate object that has performed best. May be None,
            which is equivalent to no candidate being evaluated.
        """
        return self._exp_assistants[experiment_id].get_best_candidate()

    def update(self, experiment_id, status, candidate):
        """
        Updates the specicied experiment with the status of an experiment
        evaluation.

        Parameters
        ----------
        experiment_id : string
            The id of the experiment for which to return the best candidate.
        candidate : Candidate
            The Candidate object whose status is updated.
        status : {"finished", "pausing", "working"}
            A string defining the status change. Can be one of the following:
            - finished: The Candidate is now finished.
            - pausing: The evaluation of Candidate has been paused and can be
                resumed by another worker.
            - working: The Candidate is now being worked on by a worker.

        """
        return self._exp_assistants[experiment_id].update(status=status,
                                                         candidate=candidate)

    def get_experiment_as_dict(self, exp_id):
        """
        Returns the specified experiment as dictionary.

        Parameters
        ----------
        exp_id : string
            The id of the experiment.

        Returns
        -------
        exp_dict : dict
            The experiment dictionary as defined by Experiment.to_dict().
        """
        return self._exp_assistants[exp_id].get_experiment_as_dict()

    def get_plot_result_per_step(self, exp_id):
        """
        Returns the figure for the result of each step.

        Parameters
        ----------
        exp_id : string
            The id of the experiment.

        Result
        ------
        fig : matplotlib.figure
            The figure containing the result of each step.
        """
        return self._exp_assistants[exp_id].plot_result_per_step()

    def contains_id(self, id):
        """
        Tests whether this lab assistant has an experiment with id.

        Parameters
        ----------
        id : string
            The ID to be tested.

        Returns
        -------
        contains : bool
            True iff this lab assistant contains an experiment with this id.
        """
        if id in self._exp_assistants:
            return True
        return False

    def get_ids(self):
        """
        Returns all known ids for this lab assistant.

        Returns
        -------
        ids : list of strings
            All ids this lab assitant knows.
        """
        return self._exp_assistants.keys()


    def set_exit(self):
        """
        Exits this assistant.

        Currently, all that is done is exiting all exp_assistants..
        """
        for exp in self._exp_assistants.values():
            exp.set_exit()