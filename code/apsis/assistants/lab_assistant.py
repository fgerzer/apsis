__author__ = 'Frederik Diehl'

from apsis.assistants.experiment_assistant import BasicExperimentAssistant, PrettyExperimentAssistant
import matplotlib.pyplot as plt
from apsis.utilities.plot_utils import _create_figure, _polish_figure, plot_lists, write_plot_to_file
import time
import datetime
import os

class BasicLabAssistant(object):
    """
    This is used to control multiple experiments at once.

    This is done by abstracting a dict of named experiment assistants.

    Attributes
    ----------
    exp_assistants: dict of ExperimentAssistants.
        The dictionary of experiment assistants this LabAssistant uses.
    """
    exp_assistants = None

    def __init__(self):
        """
        Initializes the lab assistant with no experiments.
        """
        self.exp_assistants = {}

    def init_experiment(self, name, optimizer, param_defs,
                        optimizer_arguments=None, minimization=True):
        """
        Initializes a new experiment.

        Parameters
        ----------
        name: string
            The name of the experiment. This has to be unique.
        optimizer: Optimizer instance or string
            This is an optimizer implementing the corresponding functions: It
            gets an experiment instance, and returns one or multiple candidates
            which should be evaluated next.
            Alternatively, it can be a string corresponding to the optimizer,
            as defined by apsis.utilities.optimizer_utils.
        param_defs: dict of ParamDef.
            This is the parameter space defining the experiment.
        optimizer_arguments=None: dict
            These are arguments for the optimizer. Refer to their documentation
            as to which are available.
        minimization=True: bool
            Whether the problem is one of minimization or maximization.
        """
        if name in self.exp_assistants:
            raise ValueError("Already an experiment with name %s registered."
                             %name)
        self.exp_assistants[name] = PrettyExperimentAssistant(name, optimizer,
            param_defs, optimizer_arguments=optimizer_arguments,
            minimization=minimization)

    def get_next_candidate(self, exp_name):
        """
        Returns the Candidate next to evaluate for a specific experiment.

        Parameters
        ----------
        exp_name: string
            Has to be in experiment_assistants.

        Returns
        -------
        next_candidate: Candidate or None:
            The Candidate object that should be evaluated next. May be None.
        """
        return self.exp_assistants[exp_name].get_next_candidate()

    def update(self, exp_name, candidate, status="finished"):
        """
        Updates the experiment with the status of an experiment
        evaluation.

        Parameters
        ----------
        exp_name: string
            Has to be in experiment_assistants
        candidate: Candidate
            The Candidate object whose status is updated.
        status=finished: string
            A string defining the status change. Can be one of the following:
            - finished: The Candidate is now finished.
            - pausing: The evaluation of Candidate has been paused and can be
                resumed by another worker.
            - working: The Candidate is now being worked on by a worker.
        """
        self.exp_assistants[exp_name].update(candidate, status=status)

    def get_best_candidate(self, exp_name):
        """
        Returns the best candidate to date for a specific experiment.

        Parameters
        ----------
        exp_name: string
            Has to be in experiment_assistants.

        Returns
        -------
        best_candidate: candidate or None
            Returns a candidate if there is a best one (which corresponds to
            at least one candidate evaluated) or None if none exists.
        """
        return self.exp_assistants[exp_name].get_best_candidate()

class PrettyLabAssistant(BasicLabAssistant):
    COLORS = ["g", "r", "c", "b", "m", "y"]

    write_directory_base = None
    lab_run_directory = None
    global_start_date = None

    def __init__(self, write_directory_base="/tmp/APSIS_WRITING"):
        super(PrettyLabAssistant, self).__init__()

        self.write_directory_base = write_directory_base
        self.global_start_date = time.time()

        self._init_directory_structure()

    def update(self, exp_name, candidate, status="finished"):
        super(PrettyLabAssistant, self).update(exp_name, candidate, status=status)

        #trigger the writing, but by default only on equal steps
        self.write_out_plots_current_step(same_steps_only=True)

    def write_out_plots_current_step(self, same_steps_only=True):
        """
        This method will write out all plots available to the path
        configured in self.lab_run_directory.

        Parameters
        ---------
        same_steps_only=True: boolean
            Write only if all experiment assistants in this lab assistant
            are currently in the same step.
        """
        step_string, same_step = self._compute_current_step_overall()
        if same_steps_only and not same_step:
            return

        plot_base = os.path.join(self.lab_run_directory, "plots")
        plot_step_base = os.path.join(plot_base, step_string)
        self._ensure_directory_exists(plot_step_base)

        #this hash will store all the plots to write
        plots_to_write = {}

        #result per step plot
        result_per_step = self.plot_result_per_step(
            experiments=self.exp_assistants.keys(),
            show_plot=False)
        plots_to_write['result_per_step'] = result_per_step

        #TODO add new plots here if any!

        #finally write out all plots created above to their files

        for plot_name in plots_to_write.keys():
            plot_fig = plots_to_write[plot_name]

            write_plot_to_file(plot_fig, plot_name + "_" + step_string, plot_step_base)

    def plot_result_per_step(self, experiments, show_plot=True, plot_at_least=1):
        """
        Returns (and plots) the plt.figure plotting the results over the steps
        for the specified experiments.

        Parameters
        ----------
        experiments: list of experiment names or experiment name.
            The experiments to plot.
        show_plot=True: bool
            Whether to show the plot after creation.
        fig=None: None or pyplot figure
            The figure to update. If None, a new figure will be created.
        color="b": string
            A string representing a pyplot color.
        plot_at_least=1: float
            The percentage of entries to show.

        Returns
        -------
        fig: plt.figure
            The figure containing the results over the steps.
        """
        if not isinstance(experiments, list):
            experiments = [experiments]

        plots_list = []
        for i, ex_name in enumerate(experiments):
            exp_ass = self.exp_assistants[ex_name]
            plots_list.extend(exp_ass._best_result_per_step_dicts(color=self.COLORS[i % len(self.COLORS)]))

        if self.exp_assistants[experiments[0]].experiment.minimization_problem:
            legend_loc = 'upper right'
            plot_min = 1
            plot_max = plot_at_least
        else:
            legend_loc = 'upper left'
            plot_min = plot_at_least
            plot_max = 1
        plot_options = {
            "legend_loc": legend_loc,
            "x_label": "steps",
            "y_label": "result",
            "title": "Comparison of %s." % experiments
        }
        fig = plot_lists(plots_list, fig_options=plot_options, plot_at_least=(plot_min, plot_max))

        if show_plot:
            plt.show(True)
            
        return fig

    def _init_directory_structure(self):
        """
        Method to create the directory structure if not exists
        for results and plots writing
        """
        if self.lab_run_directory is None:
            date_name = datetime.datetime.utcfromtimestamp(
                self.global_start_date).strftime("%Y-%m-%d_%H:%M:%S")

            self.lab_run_directory = os.path.join(self.write_directory_base,
                                                  date_name)

            plot_directory = os.path.join(self.lab_run_directory, "plots")

            #make all the dirs if not exist, directly to the plots of this run
            #will create all sub dirs
            if not os.path.exists(plot_directory):
                os.makedirs(plot_directory)

    def _ensure_directory_exists(self, directory):
        """
        Creates the given directory if not existed.
        """
        if not os.path.exists(directory):
                os.makedirs(directory)

    def _compute_current_step_overall(self):
        """
        Compute the string used to describe the current state of experiments

        If we have three running experiments in this lab assistant, then
        we can have the first in step 3, the second in step 100 and the third
        in step 1 - hence this would yield the step string "3_100_1".

        Returns
        -------

        step_string: string
            The string describing the overall steps of experiments.

        same_step: boolean
            A boolean if all experiments are in the same step.
        """

        step_string = ""
        last_step = 0
        same_step = True

        experiment_names_sorted = sorted(self.exp_assistants.keys())

        for i, ex_assistant_name in enumerate(experiment_names_sorted):
            experiment = self.exp_assistants[ex_assistant_name].experiment
            curr_step = len(experiment.candidates_finished)
            if i == 0:
                last_step = curr_step
            elif last_step != curr_step:
                same_step = False

            step_string += str(curr_step)

            if not i == len(experiment_names_sorted)  - 1:
                step_string += "_"

        return step_string, same_step