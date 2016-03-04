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
from apsis.models import experiment

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
    _write_directory_base : String, optional
        The directory to write all the results and plots to.
    _logger : logging.logger
        The logger for this class.
    """
    _exp_assistants = None

    _write_directory_base = None
    _lab_run_directory = None
    _global_start_date = None
    _logger = None



    def __init__(self, write_directory_base=None, continue_path=None):
        """
        Initializes the lab assistant.

        Parameters
        ----------
        write_directory_base : string, optional
            Sets the base write directory for the lab assistant. If None
            (default) the directory depends on the operating system.
            ./APSIS_WRITING if on Windows, /tmp/APSIS_WRITING otherwise.
        continue_path : string, optional
            The path for continuing the experiment. If None (default) a new
            lab assistant is started.
        """
        self._logger = get_logger(self)
        if write_directory_base is None:
            if os.name == "nt":
                write_directory_base = os.path.relpath("APSIS_WRITING")
            else:
                write_directory_base = "/tmp/APSIS_WRITING"

        self._logger.info("Initializing lab assistant.")
        self._logger.info("Writing results to %s" %write_directory_base)
        self._write_directory_base = write_directory_base

        self._exp_assistants = {}
        if continue_path is None:
            self._global_start_date = time.time()
            self._init_directory_structure()
        else:
            # set the correct path.
            self._lab_run_directory = continue_path
            with open(continue_path + "/lab_assistant.json", 'r') as infile:
                lab_assistant_json = json.load(infile)
            self._global_start_date = lab_assistant_json["global_start_date"]
            for p in lab_assistant_json["exp_assistants"].values():
                self._load_exp_assistant_from_path(p)

        self._write_state_to_file(self._lab_run_directory)
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

        exp_ass = ExperimentAssistant(optimizer, optimizer_arguments=optimizer_arguments,
                            write_directory_base=self._lab_run_directory,
                            csv_write_frequency=1)
        exp_ass.init_experiment(name, param_defs, exp_id, notes, minimization)
        self._exp_assistants[exp_id] = exp_ass
        self._logger.info("Experiment initialized successfully.")
        self._write_state_to_file(self._lab_run_directory)
        return exp_id

    def _init_directory_structure(self):
        """
        Method to create the directory structure if it does not exist
        for results and plot writing.
        """
        if self._lab_run_directory is None:
            date_name = datetime.datetime.utcfromtimestamp(
                self._global_start_date).strftime("%Y-%m-%d_%H.%M.%S")

            self._lab_run_directory = os.path.join(self._write_directory_base,
                                                  date_name)

            ensure_directory_exists(self._lab_run_directory)

    def _load_exp_assistant_from_path(self, path):
        with open(path + "/exp_assistant.json", 'r') as infile:
            exp_assistant_json = json.load(infile)

        optimizer_class = exp_assistant_json["optimizer_class"]
        optimizer_arguments = exp_assistant_json["optimizer_arguments"]
        write_directory_base = exp_assistant_json["write_directory_base"]
        experiment_directory = exp_assistant_json["experiment_directory"]
        csv_write_frequency = exp_assistant_json["csv_write_frequency"]

        exp_ass = ExperimentAssistant(optimizer_class, optimizer_arguments,
                                      write_directory_base, experiment_directory,
                                      csv_write_frequency)
        exp = self._load_experiment(path)
        exp_ass.set_experiment(exp)
        self._exp_assistants[exp_ass.get_exp_id()] = exp_ass

    def _load_experiment(self, path):
        with open(path + "/experiment.json", 'r') as infile:
            exp_json = json.load(infile)
        exp = experiment.from_dict(exp_json)
        return exp


    def _write_state_to_file(self, path):
        state = {"global_start_date": self._global_start_date,
                "exp_assistants": {x.get_exp_id(): x._experiment_directory_base for x
                                    in self._exp_assistants.values()}}
        with open(path + '/lab_assistant.json', 'w') as outfile:
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
        self.write_out_plots_current_step(self._exp_assistants.keys())
        return self._exp_assistants[experiment_id].update(status=status,
                                                         candidate=candidate)


    def plot_result_per_step(self, experiments, plot_min=None,
                             plot_max=None, title=None, plot_up_to=None):
        """
        Returns (and plots) the plt.figure plotting the results over the steps
        for the specified experiments.

        Parameters
        ----------
        experiments : list of experiment names or experiment name.
            The experiments to plot.
        show_plot : bool, optional
            Whether to show the plot after creation.
        fig : None or pyplot figure, optional
            The figure to update. If None, a new figure will be created.
        color : string, optional
            A string representing a pyplot color.
        plot_min : float, optional
            The smallest value to plot on the y axis.
        plot_max : float, optional
            The biggest value to plot on the y axis.
        title : string, optional
            The title for the plot. If None, one is autogenerated.

        Returns
        -------
        fig : plt.figure
            The figure containing the results over the steps.
        """
        if not isinstance(experiments, list):
            experiments = [experiments]
        if title is None:
            title = "Comparison of the results of %s." % experiments
        plots_list = []
        for i, exp_id in enumerate(experiments):
            exp_ass = self._exp_assistants[exp_id]
            plots_list.extend(exp_ass._best_result_per_step_dicts(color=COLORS[i % len(COLORS)],
                                                                  plot_up_to=plot_up_to))

        if self._exp_assistants[experiments[0]]._experiment.minimization_problem:
            legend_loc = 'upper right'
        else:
            legend_loc = 'upper left'
        plot_options = {
            "legend_loc": legend_loc,
            "x_label": "steps",
            "y_label": "result",
            "title": title,
            "minimizing": self._exp_assistants[experiments[0]]._experiment.minimization_problem
        }
        fig, ax = plot_lists(plots_list, fig_options=plot_options, plot_min=plot_min, plot_max=plot_max)

        return fig


    def generate_all_plots(self, exp_ass=None, plot_up_to=None):
        """
        Function to generate all plots available.

        Returns
        -------
        figures : dict of plt.figure
            The dict contains all plots available by this assistant. Every
            plot is keyed by an identifier.
        """
        #this dict will store all the plots to write
        plots_to_write = {}

        if exp_ass is None:
            exp_ass = self._exp_assistants.keys()

        result_per_step = self.plot_result_per_step(
            experiments=exp_ass, plot_up_to=plot_up_to)

        plots_to_write['result_per_step'] = result_per_step

        #TODO in case there is new plots in this assistant add them here.

        return plots_to_write

    def _get_min_step(self):
        min_step = min([len(x._experiment.candidates_finished) for x in
                        self._exp_assistants.values()])
        return min_step

    def write_out_plots_current_step(self, exp_ass=None, same_steps_only=True):
        """
        This method will write out all plots available to the path
        configured in self.lab_run_directory.

        Parameters
        ---------
        exp_ass : list, optional
            List of experiment assistant names to include in the plots. Defaults to
            None, which is equivalent to all.
        same_steps_only : boolean, optional
            Write only if all experiment assistants in this lab assistant
            are currently in the same step.
        """
        min_step = self._get_min_step()
        if same_steps_only:
            plot_up_to = min_step
        else:
            plot_up_to = None

        plot_base = os.path.join(self._lab_run_directory, "plots")
        plot_step_base = os.path.join(plot_base, "step_" + str(min_step))
        ensure_directory_exists(plot_step_base)

        if exp_ass is None:
            exp_ass = self._exp_assistants.keys()

        plots_to_write = self.generate_all_plots(exp_ass, plot_up_to)


        #finally write out all plots created above to their files
        for plot_name in plots_to_write.keys():
            plot_fig = plots_to_write[plot_name]

            write_plot_to_file(plot_fig, plot_name + "_step" + str(min_step),
                               plot_step_base)
            plt.close(plot_fig)


    def _compute_current_step_overall(self):
        """
        Compute the string used to describe the current state of experiments
        If we have three running experiments in this lab assistant, then
        we can have the first in step 3, the second in step 100 and the third
        in step 1 - hence this would yield the step string "3_100_1".

        Returns
        -------
        step_string : string
            The string describing the overall steps of experiments.
        same_step : boolean
            A boolean if all experiments are in the same step.
        """

        step_string = ""
        last_step = 0
        same_step = True

        experiment_names_sorted = sorted(self._exp_assistants.keys())

        for i, ex_assistant_name in enumerate(experiment_names_sorted):
            experiment = self._exp_assistants[ex_assistant_name]._experiment

            curr_step = len(experiment.candidates_finished)
            if i == 0:
                last_step = curr_step
            elif last_step != curr_step:
                same_step = False

            step_string += str(curr_step)

            if not i == len(experiment_names_sorted)  - 1:
                step_string += "_"

        return step_string, same_step

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