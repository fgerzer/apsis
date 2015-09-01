__author__ = 'Frederik Diehl'

from apsis.assistants.experiment_assistant import ExperimentAssistant
from apsis.utilities.file_utils import ensure_directory_exists
import time
import datetime
import os
from apsis.utilities.logging_utils import get_logger
from apsis.utilities.plot_utils import plot_lists, write_plot_to_file
import matplotlib.pyplot as plt
import uuid
import copy
import numpy as np

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
    exp_assistants = None

    _write_directory_base = None
    _lab_run_directory = None
    _global_start_date = None
    _logger = None



    def __init__(self, write_directory_base=None):
        """
        Initializes the lab assistant.

        Parameters
        ----------
        write_directory_base : string, optional
            Sets the base write directory for the lab assistant. If None
            (default) the directory depends on the operating system.
            ./APSIS_WRITING if on Windows, /tmp/APSIS_WRITING otherwise.
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
        self._global_start_date = time.time()
        self._init_directory_structure()
        self.exp_assistants = {}
        self._logger.info("lab assistant successfully initialized.")

    def init_experiment(self, name, optimizer, param_defs, exp_id=None, notes=None,
                        optimizer_arguments=None, minimization=True):
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
        if exp_id in self.exp_assistants.keys():
            raise ValueError("Already an experiment with id %s registered."
                             %exp_id)

        if exp_id is None:
            while True:
                exp_id = uuid.uuid4().hex
                if exp_id not in self.exp_assistants.keys():
                    break

        exp_ass = ExperimentAssistant(optimizer, optimizer_arguments=optimizer_arguments,
                            write_directory_base=self._lab_run_directory,
                            csv_write_frequency=1)
        exp_ass.init_experiment(name, param_defs, exp_id, notes, minimization)
        self.exp_assistants[exp_id] = exp_ass
        self._logger.info("Experiment initialized successfully.")
        return exp_id

    def _init_directory_structure(self):
        """
        Method to create the directory structure if it does not exist
        for results and plot writing.
        """
        if self._lab_run_directory is None:
            date_name = datetime.datetime.utcfromtimestamp(
                self._global_start_date).strftime("%Y-%m-%d_%H:%M:%S")

            self._lab_run_directory = os.path.join(self._write_directory_base,
                                                  date_name)

            ensure_directory_exists(self._lab_run_directory)

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
        return self.exp_assistants[experiment_id].get_candidates()

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
        return self.exp_assistants[experiment_id].get_next_candidate()

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
        return self.exp_assistants[experiment_id].get_best_candidate()

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
        self.write_out_plots_current_step(self.exp_assistants.keys())
        return self.exp_assistants[experiment_id].update(status=status,
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
            exp_ass = self.exp_assistants[exp_id]
            plots_list.extend(exp_ass._best_result_per_step_dicts(color=COLORS[i % len(COLORS)],
                                                                  plot_up_to=plot_up_to))

        if self.exp_assistants[experiments[0]]._experiment.minimization_problem:
            legend_loc = 'upper right'
        else:
            legend_loc = 'upper left'
        plot_options = {
            "legend_loc": legend_loc,
            "x_label": "steps",
            "y_label": "result",
            "title": title
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
            exp_ass = self.exp_assistants.keys()

        result_per_step = self.plot_result_per_step(
            experiments=exp_ass, plot_up_to=plot_up_to)

        plots_to_write['result_per_step'] = result_per_step

        #TODO in case there is new plots in this assistant add them here.

        return plots_to_write

    def _get_min_step(self):
        min_step = min([len(x._experiment.candidates_finished) for x in self.exp_assistants.values()])
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
            exp_ass = self.exp_assistants.keys()

        plots_to_write = self.generate_all_plots(exp_ass, plot_up_to)


        #finally write out all plots created above to their files
        for plot_name in plots_to_write.keys():
            plot_fig = plots_to_write[plot_name]

            write_plot_to_file(plot_fig, plot_name + "_step" + str(min_step), plot_step_base)
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

        experiment_names_sorted = sorted(self.exp_assistants.keys())

        for i, ex_assistant_name in enumerate(experiment_names_sorted):
            experiment = self.exp_assistants[ex_assistant_name]._experiment

            curr_step = len(experiment.candidates_finished)
            if i == 0:
                last_step = curr_step
            elif last_step != curr_step:
                same_step = False

            step_string += str(curr_step)

            if not i == len(experiment_names_sorted)  - 1:
                step_string += "_"

        return step_string, same_step


    def set_exit(self):
        """
        Exits this assistant.

        Currently, all that is done is exiting all exp_assistants..
        """
        for exp in self.exp_assistants.values():
            exp.set_exit()

class ValidationLabAssistant(LabAssistant):
    """
    This Lab Assistant is used for validating optimization with cross-validation.
    This is done by internally multiplexing each experiment into cv many.
    Attributes
    ----------
    cv : int
        The number of crossvalidations used.
    exp_current : dict
        A dictionary of string keys and int or None values, which stores the
        last experiment for each experiment name from which a candidate has
        been returned.
    disable_auto_plot: bool
        To disable automatic plot writing functionality completely.
    """


    cv = None
    candidates_pending = None
    disable_auto_plot = None

    def __init__(self, cv=5, disable_auto_plot=False):
        """
        Initializes the ValidationLabAssistant.
        Paramters
        ---------
        cv : int
            The number of crossvalidations used.
        disable_auto_plot: bool, optional
            To disable automatic plot writing functionality completely.
        """
        super(ValidationLabAssistant, self).__init__()
        self.cv = cv
        self.disable_auto_plot = disable_auto_plot
        self.candidates_pending = {}

    def init_experiment(self, name, optimizer, param_defs, exp_id=None,
                        notes=None, optimizer_arguments=None, minimization=True):
        """
        Initializes a new experiment.
        This actually initializes self.cv many experiments.
        Internally, the experiments are called name_i, where name is the
        experiment_name and i is the number of the experiment.

        Parameters
        ----------
        name : string
            The name of the experiment. This has to be unique.
        optimizer : Optimizer instance or string
            This is an optimizer implementing the corresponding functions: It
            gets an experiment instance, and returns one or multiple candidates
            which should be evaluated next.
            Alternatively, it can be a string corresponding to the optimizer,
            as defined by apsis.utilities.optimizer_utils.
        param_defs : dict of ParamDef.
            This is the parameter space defining the experiment.
        optimizer_arguments : dict, optional
            These are arguments for the optimizer. Refer to their documentation
            as to which are available.
        minimization : bool, optional
            Whether the problem is one of minimization or maximization.
        """
        self._logger.info("Initializing new experiment \"%s\". "
                     " Parameter definitions: %s. Minimization is %s"
                     %(name, param_defs, minimization))
        if exp_id in self.exp_assistants:
            raise ValueError("Already an experiment with id %s registered."
                             %name)
        if exp_id is None:
            while True:
                exp_id = uuid.uuid4().hex
                if exp_id not in self.exp_assistants.keys():
                    break
        self.exp_assistants[exp_id] = []
        self.candidates_pending[exp_id] = []
        for i in range(self.cv):
            exp_ass = ExperimentAssistant(optimizer, optimizer_arguments=optimizer_arguments,
                                     write_directory_base=self._lab_run_directory,
                                     csv_write_frequency=1)
            exp_ass.init_experiment(name + "_" + str(i), param_defs, exp_id,
                                    notes, minimization)
            self.exp_assistants[exp_id].append(exp_ass)
            self.candidates_pending[exp_id].append([])
        self._logger.info("Experiment initialized successfully.")
        return exp_id

    def clone_experiments_by_id(self, exp_id, optimizer,
                                  optimizer_arguments, new_exp_name):
        """
        Take an existing experiment managed by this lab assistant,
        fully clone it and store it under a new name to use it with a new
        optimizer. This functionality can be used to initialize several experiments
        of several optimizers with the same points.
        For the given exp_name all underlying experiment instances are cloned and renamed.
        Then a new experiment assistant is instantiated given the cloned and renamed
        experiment using the given optimizer. The new experiment assistants are stored
        and managed inside this lab assistant. The old experiment is not touched
        and continues to be part of this lab assistant.
        The parameter definitions and other experiment specific configuration is
        copied over from the old to the new experiment.

        Parameters
        ----------
        exp_id : string
            The id of the experiment to be cloned.
        new_exp_name: string, optional
            The name the cloned experiment will have after creation. If None,
            the old name is reused.
        optimizer : Optimizer instance or string
            This is an optimizer implementing the corresponding functions: It
            gets an experiment instance, and returns one or multiple candidates
            which should be evaluated next.
            Alternatively, it can be a string corresponding to the optimizer,
            as defined by apsis.utilities.optimizer_utils.
        optimizer_arguments : dict, optional
            These are arguments for the optimizer. Refer to their documentation
            as to which are available.
        """
        self.exp_assistants[new_exp_name] = []

        while True:
            new_exp_id = uuid.uuid4().hex
            if new_exp_id not in self.exp_assistants.keys():
                break

        #every experiment has self.cv many assistants
        for i in range(len(self.exp_assistants[exp_id])):
            old_exp_assistant = self.exp_assistants[exp_id][i]
            old_exp = old_exp_assistant._experiment

            #clone and rename experiment
            new_exp = old_exp.clone()

            new_name_cved = new_exp_name + "_" + str(i)
            new_exp.name = new_name_cved

            #recreate exp assistant
            new_exp_assistant = ExperimentAssistant(optimizer, optimizer_arguments=optimizer_arguments,
                                     write_directory_base=self._lab_run_directory,
                                     csv_write_frequency=1)
            new_exp_assistant.set_experiment(new_exp)
            self.exp_assistants[new_exp_id].append(new_exp_assistant)

        self.candidates_pending[new_exp_id] = copy.deepcopy(self.candidates_pending[exp_id])

        self._logger.info("Experiment " + str(exp_id) + " cloned to " +
                         str(new_exp_id) + " and successfully initialized.")
        return new_exp_id

    def _get_min_step(self):
        #min_step = min([len(x._experiment.candidates_finished) for x in self.exp_assistants.values()])
        min_step = None
        for e_list in self.exp_assistants.values():
            for e_ass in e_list:
                if min_step is None or len(e_ass._experiment.candidates_finished) < min_step:
                    min_step = len(e_ass._experiment.candidates_finished)
        return min_step

    def update(self, exp_id, status, candidate):
        """
        Updates the experiment with a new candidate.
        This is done by updating the experiment from which the last candidate
        has been returned using get_next_candidate.
        Note that this LabAssistant does not feature the ability to update with
        arbitrary candidates.
        """
        cand_id = candidate.id
        idx = None
        for i in range(len(self.exp_assistants[exp_id])):
            if cand_id in self.candidates_pending[exp_id][i]:
                internal_idx = self.candidates_pending[exp_id][i].index(cand_id)
                del(self.candidates_pending[exp_id][i][internal_idx])
                idx = i
                break
        if idx is None:
            raise ValueError("No candidate given to the outside for that experiment.")
        self.exp_assistants[exp_id][idx].update(candidate, status)
        if not self.disable_auto_plot:
            self.write_out_plots_current_step()

    def get_next_candidate(self, exp_id):
        """
        Returns the Candidate next to evaluate for a specific experiment.
        This is done by using the get_next_candidate function from the
        sub-experiment with the least finished and pending candidates.

        Parameters
        ----------
        exp_name : string
            Has to be in experiment_assistants.
        Returns
        -------
        next_candidate : Candidate or None:
            The Candidate object that should be evaluated next. May be None.
        """
        index_min_finished = 0
        num_min_finished = len(self.exp_assistants[exp_id][0]._experiment.candidates_finished) \
                              + len(self.candidates_pending[exp_id][0])
        for i in range(len(self.exp_assistants[exp_id])):
            fin_and_pending = len(self.exp_assistants[exp_id][i]._experiment.candidates_finished) \
                              + len(self.candidates_pending[exp_id][i])
            if num_min_finished > fin_and_pending:
                index_min_finished = i
                num_min_finished = fin_and_pending
        cand = self.exp_assistants[exp_id][index_min_finished].get_next_candidate()
        if cand is not None:
            self.candidates_pending[exp_id][index_min_finished].append(cand.id)
        return cand

    def plot_result_per_step(self, experiments, show_plot=False, plot_min=None, plot_max=None, title=None):
        """
        Returns (and plots) the plt.figure plotting the results over the steps
        for the specified experiments.
        This includes:
            - one dot per evaluated result at a step
            - a line showing the best result found up to that step for every step
            - error bars for that line

        Parameters
        ----------
        experiments : list of experiment names or experiment name.
            The experiments to plot.
        show_plot : bool, optional
            Whether to show the plot after creation.
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

        best_per_step_plots_list, step_plots_list, plot_options = self._gen_plot_data(experiments, title)
        plots_list = []
        plots_list.extend(best_per_step_plots_list)
        plots_list.extend(step_plots_list)
        fig, ax = plot_lists(plots_list, fig_options=plot_options, plot_min=plot_min, plot_max=plot_max)

        if show_plot:
            plt.show(True)

        return fig

    def plot_validation(self, experiments, show_plot=True, plot_min=None, plot_max=None, title=None):
        """
        Returns (and plots) the plt.figure plotting the results over the steps
        for the specified experiments.
        This includes:
            - a line showing the best result found up to that step for every step
            - error bars for that line

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
        best_per_step_plots_list, step_plots_list, plot_options = self._gen_plot_data(experiments)
        plots_list = []
        plots_list.extend(best_per_step_plots_list)

        fig, ax = plot_lists(plots_list, fig_options=plot_options, plot_min=plot_min, plot_max=plot_max)

        if show_plot:
            plt.show(True)

        return fig

    def get_best_candidate(self, exp_name):
        """
        Returns the best candidate to date for a specific experiment.
        The best candidate is the best candidate from all of the experiments.

        Parameters
        ----------
        exp_name : string
            Has to be in experiment_assistants.
        Returns
        -------
        best_candidate : candidate or None
            Returns a candidate if there is a best one (which corresponds to
            at least one candidate evaluated) or None if none exists.
        """
        best_candidate = self.exp_assistants[exp_name][0].get_best_candidate()
        for c in self.get_best_candidates(exp_name):
            if self.exp_assistants[exp_name][0]._experiment.better_cand(c, best_candidate):
                best_candidate = c
        return best_candidate

    def get_best_candidates(self, exp_name):
        """
        Returns the best candidates to date for each crossvalidated experiment.
        This is a list of candidates, on which further statistics like mean
        and variance can be computed.

        Parameters
        ----------
        exp_name : string
            Has to be in experiment_assistants.
        Returns
        -------
        best_candidates : list of candidates or None
            For each subexperiment, returns a candidate if there is a best one
            (which corresponds to at least one candidate evaluated) or None
            if none exists.
        """
        best = []
        for e in self.exp_assistants[exp_name]:
            best.append(e.get_best_candidate())
        return best

    def _gen_plot_data(self, experiments, title=None):
        """
        Generates plot data for use with plot_validation and
        plot_best_result_per_step

        Parameters
        ----------
        experiments : (List of) experiments
            The experiments to plot
        title : string, optional
            The title for the plot. If None, one is autogenerated.
        Returns
        -------
        best_per_step_plots_list : List of dict
            A list of the best result at each step, for use with plot_utils
        step_plots_list : List of dicts
            A list of the evaluated result at each step, for use with plot_utils
        plot_options : dict
            Options for the plot, for use with plot_utils

        """
        #TODO document the matplotlib hack.

        if not isinstance(experiments, list):
            experiments = [experiments]
        if title is None:
            title = "Comparison of the results of %s." % [self.exp_assistants[x][0]._experiment.name for x in experiments]

        best_per_step_plots_list = []
        step_plots_list = []
        for i, exp_id in enumerate(experiments):
            step_dicts = []
            best_vals = []
            for c in range(self.cv):
                exp_ass = self.exp_assistants[exp_id][c]
                cur_step_dict, best_dict = exp_ass._best_result_per_step_dicts(color=COLORS[i % len(COLORS)])
                cur_step_dict["label"] = None
                step_dicts.append(cur_step_dict)
                best_vals.append(best_dict["y"])
            best_mean = []
            best_var = []
            for i in range(max([len(x) for x in best_vals])):
                vals = []
                for c in range(self.cv):
                    if len(best_vals[c]) > i:
                        vals.append(best_vals[c][i])
                best_mean.append(np.mean(vals))
                best_var.append((np.var(vals))**0.5 * 1.15) #75% confidence
            x = [x for x in range(len(best_mean))]
            best_dict["y"] = best_mean
            best_dict["var"] = best_var
            best_dict["x"] = x
            best_dict["label"] = exp_ass._experiment.name
            # This is necessary to avoid a matplotlib crash.
            if len(best_var) > 0:
                best_per_step_plots_list.append(best_dict)
                step_plots_list.extend(step_dicts)

        if self.exp_assistants[experiments[0]][0]._experiment.minimization_problem:
            legend_loc = 'upper right'
        else:
            legend_loc = 'upper left'
        plot_options = {
            "legend_loc": legend_loc,
            "x_label": "steps",
            "y_label": "result",
            "title": title
        }
        return best_per_step_plots_list, step_plots_list, plot_options

    def _compute_current_step_overall(self):
        """
        Compute the string used to describe the current state of experiments
        If we have three running experiments in this lab assistant, then
        we can have the first in step 3, the second in step 100 and the third
        in step 1 - hence this would yield the step string "3_100_1".
        The step of the crossvalidated experiments is the minimum step each
        of them has achieved.
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

        experiment_names_sorted = sorted(self.exp_assistants.keys())

        for i, ex_assistant_name in enumerate(experiment_names_sorted):
            exp_asss = self.exp_assistants[ex_assistant_name]
            curr_step = len(exp_asss[0]._experiment.candidates_finished)

            for e in exp_asss:
                curr_step = min(curr_step, e._experiment.candidates_finished)
            if i == 0:
                last_step = curr_step
            elif last_step != curr_step:
                same_step = False

            step_string += str(curr_step)

            if not i == len(experiment_names_sorted)  - 1:
                step_string += "_"

        return step_string, same_step

    def generate_all_plots(self, exp_ass=None, plot_up_to=None):
        """
        Function to generate all plots available.
        Returns
        -------
        figures : dict of plt.figure
            The hash contains all plots available by this assistant. Every
            plot is keyed by an identifier.
        """
        #this hash will store all the plots to write
        plots_to_write = {}
        if exp_ass is None:
            exp_ass = self.exp_assistants.keys()

        result_per_step = self.plot_result_per_step(
            experiments=exp_ass,
            show_plot=False)

        plots_to_write['result_per_step'] = result_per_step

        plot_validation = self.plot_validation(
            experiments=self.exp_assistants.keys(), show_plot=False)
        plots_to_write['validation'] = plot_validation

        return plots_to_write

    def set_exit(self):
        """
        Exits this assistant.

        Currently, all that is done is exiting all exp_assistants..
        """
        for exp_list in self.exp_assistants.values():
            for exp in exp_list:
                exp.set_exit()