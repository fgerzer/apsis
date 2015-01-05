__author__ = 'Frederik Diehl'

from apsis.assistants.experiment_assistant import BasicExperimentAssistant, PrettyExperimentAssistant
import matplotlib.pyplot as plt
from apsis.utilities.plot_utils import _create_figure, _polish_figure, plot_lists

#TODO document.
class BasicLabAssistant(object):
    exp_assistants = None

    def __init__(self):
        self.exp_assistants = {}

    def init_experiment(self, name, optimizer, param_defs,
                        optimizer_arguments=None, minimization=True):
        if name in self.exp_assistants:
            raise ValueError("Already an experiment with name %s registered."
                             %name)
        self.exp_assistants[name] = PrettyExperimentAssistant(name, optimizer,
            param_defs, optimizer_arguments=optimizer_arguments,
            minimization=minimization)

    def get_next_candidate(self, exp_name):
        return self.exp_assistants[exp_name].get_next_candidate()

    def update(self, exp_name, candidate, status="finished"):
        self.exp_assistants[exp_name].update(candidate, status=status)

    def get_best_candidate(self, exp_name):
        return self.exp_assistants[exp_name].get_best_candidate()

class PrettyLabAssistant(BasicLabAssistant):
    COLORS = ["g", "r", "c", "b", "m", "y"]

    def plot_result_per_step(self, experiments, show_plot=True, plot_at_least=1):

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