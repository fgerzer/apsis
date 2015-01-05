__author__ = 'Frederik Diehl'

from apsis.assistants.experiment_assistant import BasicExperimentAssistant, PrettyExperimentAssistant
import matplotlib.pyplot as plt
from apsis.utilities.plot_utils import _create_figure, _polish_figure, plot_lists


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
        plt.show(True)