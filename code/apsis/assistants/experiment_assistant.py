__author__ = 'Frederik Diehl'

from apsis.models.experiment import Experiment
from apsis.models.candidate import Candidate
from apsis.utilities.optimizer_utils import check_optimizer
from apsis.utilities.file_utils import ensure_directory_exists
import datetime
import os
import time
from apsis.utilities.logging_utils import get_logger
from apsis.utilities.plot_utils import plot_lists, write_plot_to_file
import matplotlib.pyplot as plt
import json

AVAILABLE_STATUS = ["finished", "pausing", "working"]


class ExperimentAssistant(object):
    """
    This class represents an assistant assisting with a single experiment.

    It stores an experiment (a data structure which stores the parameter
    definition, the candidates evaluated and current and whether it's to be
    minimized) and an optimizer for optimizing the experiment.
    It also contains functions for writing out results and for plotting.

    Parameters
    ----------
    _optimizer : Optimizer
        The Optimizer used to find new points for the experiment. It has to be
        an apsis.optimizers.optimizer.Optimizer instance.
    _optimizer_arguments : dict
        Dictionary of the arguments for optimizer.
    _experiment : Experiment
        The experiment storing the evaluated points and parameter definition.
    _write_dir : basestring
        Directory containing the checkpoints.
    _logger : logger
        The logger instance for this class.
    """

    _optimizer = None
    _optimizer_arguments = None
    _experiment = None

    _write_dir = None

    _logger = None

    def __init__(self, optimizer_class, experiment,
                 optimizer_arguments=None,
                 write_dir=None):
        """
        Initializes this experiment assistant.

        Note that calling this function does not yet create an experiment, for
        that, use init_experiment. If there is an already existing experiment,
        you can just set self._experiment.

        Parameters
        ----------
        optimizer_class : subclass of Optimizer
            The class of the optimizer, used to initialize it.
        experiment : Experiment
            The experiment representing the data of this experiment assistant.
        write_dir : basestring, optional
            The directory the state of this experiment assistant is regularly
            written to. If this is None (default), no state will be written.
        optimizer_arguments : dict, optional
            The dictionary of optimizer arguments. If None, default values will
            be used.
        """
        self._logger = get_logger(self, extra_info="exp_id: " +
                                                   str(experiment.exp_id))
        self._logger.info("Initializing experiment assistant.")
        self._optimizer = optimizer_class
        self._optimizer_arguments = optimizer_arguments
        self._write_dir = write_dir
        self._experiment = experiment
        self._init_optimizer()
        self._write_state_to_file()
        self._logger.info("Experiment assistant successfully initialized.")

    def _init_optimizer(self):
        """
        Initializes the optimizer if it does not exist.
        """
        self._logger.debug("Initializing optimizer. Current state is %s"
                           %self._optimizer)
        self._optimizer= check_optimizer(self._optimizer, self._experiment,
            optimizer_arguments=self._optimizer_arguments)
        self._logger.debug("Initialized optimizer. State afterwards is %s"
                           %self._optimizer)

    def get_next_candidate(self):
        """
        Returns the Candidate next to evaluate.

        Internally, it first tries to return the most recent pending candidate
        of this experiment. If there is none, it generates one from optimizer.

        Returns
        -------
        next_candidate : Candidate or None
            The Candidate object that should be evaluated next. May be None,
            which is equivalent to no candidate generated.
        """

        self._logger.debug("Returning next candidate.")
        to_return = None
        if not self._experiment.candidates_pending:
            self._logger.debug("No candidate pending; requesting one from "
                               "optimizer.")
            candidates = self._optimizer.get_next_candidates(num_candidates=1)
            self._logger.debug("Got %s", [str(c) for c in candidates])
            if candidates is None:
                to_return = None
            elif len(candidates) > 0:
                self._experiment.add_working(candidates[0])
                to_return = candidates[0]
        else:
            self._logger.debug("Had at least one pending.")
            cand = self._experiment.candidates_pending.pop()
            self._experiment.add_working(cand)
            to_return = cand
        self._logger.debug("Returning candidate %s" %str(to_return))
        self._write_state_to_file()
        return to_return

    def get_experiment_as_dict(self):
        """
        Returns the dictionary describing this EAss' experiment.

        Signature is equivalent to Experiment.to_dict()

        Returns
        -------
            exp_dict : dict
                The experiment dictionary.
        """
        self._logger.debug("Returning experiment as dict.")
        exp_dict = self._experiment.to_dict()
        self._logger.debug("Exp_dict is %s" %exp_dict)
        return exp_dict

    def update(self, candidate, status="finished"):
        """
        Updates the experiment_assistant with the status of an experiment
        evaluation.

        Parameters
        ----------
        candidate : Candidate
            The Candidate object whose status is updated.
        status : {"finished", "pausing", "working"}
            A string defining the status change. Can be one of the following:
            - finished: The Candidate is now finished.
            - pausing: The evaluation of Candidate has been paused and can be
                resumed by another worker.
            - working: The Candidate is now being worked on by a worker.

        """
        self._logger.debug("Updating experiment assistant with candidate %s,"
                           "status %s" %(candidate, status))
        if status not in AVAILABLE_STATUS:
            message = ("status not in %s but %s."
                             %(str(AVAILABLE_STATUS), str(status)))
            self._logger.error(message)
            raise ValueError(message)

        if not isinstance(candidate, Candidate):
            message = ("candidate %s not a Candidate instance."
                             %str(candidate))
            self._logger.error(message)
            raise ValueError(message)

        self._logger.debug("Got new %s of candidate %s with parameters %s"
                         " and result %s", status, candidate, candidate.params,
                          candidate.result)

        if status == "finished":
            self._experiment.add_finished(candidate)
            self._logger.debug("Was finished, updating optimizer.")
            # And we rebuild the new optimizer.
            self._optimizer.update(self._experiment)
            self._logger.debug("Optimizer updated.")
        elif status == "pausing":
            self._experiment.add_pausing(candidate)
        elif status == "working":
            self._experiment.add_working(candidate)
        self._write_state_to_file()

    def _write_state_to_file(self):
        """
        Writes the current state to the specified file.

        When this is called, it collects the state of this experiment assistant
        - that is, optimizer_class, optimizer_arguments and write_dir - and
        writes them to file. It also forces _experiment to write its state to
        file.
        All of this only happens if _write_dir is not None - if it is, we will
        do nothing.
        """
        self._logger.debug("Writing experiment assistant status to file %s",
                           self._write_dir)
        if self._write_dir is None:
            self._logger.debug("No write directory is set; not writing "
                               "anything.")
            return
        state = {}
        opt = self._optimizer
        if not isinstance(opt, basestring):
            opt = opt.name
        state["optimizer_class"] = opt
        state["optimizer_arguments"] = self._optimizer_arguments
        state["write_dir"] = self._write_dir
        with open(self._write_dir + '/exp_assistant.json', 'w') as outfile:
            json.dump(state, outfile)
        self._logger.debug("Writing state %s", state)
        self._experiment.write_state_to_file(self._write_dir)

    def get_best_candidate(self):
        """
        Returns the best candidate to date.

        Returns
        -------
        best_candidate : candidate or None
            Returns a candidate if there is a best one (which corresponds to
            at least one candidate evaluated) or None if none exists.
        """
        self._logger.debug("Returning best candidate.")
        best_candidate = self._experiment.best_candidate
        self._logger.debug("Best candidate is %s", best_candidate)
        return best_candidate

    def _best_result_per_step_dicts(self, color="b", plot_up_to=None,
                                    cutoff_percentage=1.):
        """
        Returns a dict to use with plot_utils.
        Parameters
        ----------
        color="b": string
            A pyplot-color representing string. Both plots will have that
            color.
        Returns
        -------
        dicts: list of dicts
            Two dicts, one for step_eval, one for step_best, and their
            corresponding definitions.
        """
        self._logger.debug("Returning best result per step dicts.")
        x, step_eval, step_best = self._best_result_per_step_data(plot_up_to=
                                                                  plot_up_to)

        step_eval_dict = {
            "x": x,
            "y": step_eval,
            "type": "scatter",
            "label": "%s" % (str(self._experiment.name)),
            "color": color,
            "cutoff_percent": cutoff_percentage
        }

        step_best_dict = {
            "x": x,
            "y": step_best,
            "type": "line",
            "color": color,
        }
        result = [step_eval_dict, step_best_dict]
        self._logger.debug("Returning %s", result)
        return result

    def _best_result_per_step_data(self, plot_up_to=None):
        """
        This internal function returns quality of the results by step.
        This returns an x coordinate, and for each of them a value for the
        currently evaluated result and the best found result.
        Returns
        -------
        x: list of ints
            The results per step. Should usually be [0, ..., maxSteps]
        step_evaluation: list of floats
            The result of the evaluated candidate during the corresponding step
        step_best: list of floats
            The best result that has been found until then.
        """
        self._logger.debug("Returning best result per step dicts.")
        x = []
        step_evaluation = []
        step_best = []
        best_candidate = None
        if plot_up_to is None:
            plot_up_to = len(self._experiment.candidates_finished)
        self._logger.debug("Plotting %s candidates", plot_up_to)
        for i, e in enumerate(self._experiment.candidates_finished
                              [:plot_up_to]):
            x.append(i)
            step_evaluation.append(e.result)
            if self._experiment.better_cand(e, best_candidate):
                best_candidate = e
                step_best.append(e.result)

            else:
                step_best.append(best_candidate.result)
        self._logger.debug("Returning x: %s, step_eval: %s and step_best %s",
                           x, step_evaluation, step_best)
        return x, step_evaluation, step_best

    def get_candidates(self):
        """
        Returns the candidates of this experiment in a dict.

        Returns
        -------
        result : dict
            A dictionary of three lists with the keys finished, pending and
            working, with the corresponding candidates.
        """
        self._logger.debug("Returning candidates of exp_ass.")
        result = {"finished": self._experiment.candidates_finished,
                  "pending": self._experiment.candidates_pending,
                  "working": self._experiment.candidates_working}
        self._logger.debug("Candidates are %s", result)
        return result

    def plot_result_per_step(self, ax=None, color="b",
                             plot_min=None, plot_max=None):
        """
        Returns the plt.figure plotting the results over the steps.

        Parameters
        ----------
        ax : None or matplotlib.Axes, optional
            The ax to update. If None, a new figure will be created.
        color : string, optional
            A string representing a pyplot color.
        plot_min : float, optional
            The smallest value to plot on the y axis.
        plot_max : float, optional
            The biggest value to plot on the y axis.
        Returns
        -------
        ax: plt.Axes
            The Axes containing the results over the steps.
        """
        self._logger.debug("Plotting result per step. ax %s, colors %s, "
                           "plot_min %s, plot_max %s", ax, color, plot_min,
                           plot_max)
        plots = self._best_result_per_step_dicts(color, cutoff_percentage=0.5)
        if self._experiment.minimization_problem:
            legend_loc = 'upper right'
        else:
            legend_loc = 'upper left'
        self._logger.debug("Setting legend to %s LOC", legend_loc)
        plot_options = {
            "legend_loc": legend_loc,
            "x_label": "steps",
            "y_label": "result",
            "title": "Plot of %s result over the steps."
                     % (str(self._experiment.name)),
            "minimizing": self._experiment.minimization_problem
        }
        self._logger.debug("Plot options are %s", plot_options)
        fig, ax = plot_lists(plots, ax=ax, fig_options=plot_options,
                             plot_min=plot_min, plot_max=plot_max)

        return fig

    def set_exit(self):
        """
        Exits this assistant.

        Currently, all that is done is that the optimizer is exited.
        """
        self._logger.debug("Exp assistant received exit.")
        self._optimizer.exit()
        self._logger.debug("Sent exit to optimizer.")

    @property
    def exp_id(self):
        self._logger.debug("Returning exp id.")
        exp_id = self._experiment.exp_id
        self._logger.debug("Exp_id is %s", exp_id)
        return exp_id

    @property
    def write_dir(self):
        self._logger.debug("Returning write_dir")
        write_dir = self._write_dir
        self._logger.debug("write_dir is %s", write_dir)
        return write_dir
