__author__ = 'Frederik Diehl'

from apsis.models.experiment import Experiment
from apsis.models.candidate import Candidate
from apsis.utilities.optimizer_utils import check_optimizer
from apsis.utilities.file_utils import ensure_directory_exists
import datetime
import os
import time
from apsis.utilities.logging_utils import get_logger
import signal
import multiprocessing
import Queue
from apsis.utilities.plot_utils import plot_lists, write_plot_to_file
import matplotlib.pyplot as plt

AVAILABLE_STATUS = ["finished", "pausing", "working"]

class ExperimentAssistant():

    _optimizer = None
    _optimizer_arguments = None
    _experiment = None

    _optimizer_queue = None
    _optimizer_in_queue = None
    _optimizer_process = None

    _write_directory_base = None
    _csv_write_frequency = None
    _csv_steps_written = 0
    _experiment_directory_base = None

    _logger = None

    def __init__(self, name, optimizer, param_defs, experiment=None,
                 optimizer_arguments=None, minimization=True,
                 write_directory_base="/tmp/APSIS_WRITING",
                 experiment_directory_base=None, csv_write_frequency=1):
        self._logger = get_logger(self)
        self._logger.info("Initializing experiment assistant.")
        self._optimizer = optimizer
        self._optimizer_arguments = optimizer_arguments
        if experiment is None:
            experiment = Experiment(name, param_defs, minimization)
        self._experiment = experiment
        self._csv_write_frequency = csv_write_frequency
        if self._csv_write_frequency != 0:
            self._write_directory_base = write_directory_base
            if experiment_directory_base is not None:
                self._experiment_directory_base = experiment_directory_base
                ensure_directory_exists(self._experiment_directory_base)
            else:
                self._create_experiment_directory()
        self._build_new_optimizer()
        self._logger.info("Experiment assistant for %s successfully "
                         "initialized." %name)

    def get_next_candidate(self):
        """
        Returns the Candidate next to evaluate.

        Returns
        -------
        next_candidate : Candidate or None
            The Candidate object that should be evaluated next. May be None.
        """

        self._logger.info("Returning next candidate.")
        if not self._experiment.candidates_pending:
            try:
                new_candidate = self._optimizer_queue.get_nowait()
                self._experiment.candidates_pending.append(
                    new_candidate
                )
            except Queue.Empty:
                return None
        else:
            return self._experiment.candidates_pending.pop()
        return None

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

        self._logger.info("Got new %s of candidate %s with parameters %s"
                         " and result %s" %(status, candidate, candidate.params,
                                            candidate.result))

        if status == "finished":
            self._experiment.add_finished(candidate)

            #invoke the writing to files
            step = len(self._experiment.candidates_finished)
            if self._csv_write_frequency != 0 and step != 0 \
                    and step % self._csv_write_frequency == 0:
                self._append_to_detailed_csv()
                self.write_plots()
            # And we rebuild the new optimizer.
            self._optimizer_in_queue.put(self._experiment)
            #TODO Commenting out the below means we cannot kill the optimizer
            # during optimization, even when we have new information. On the
            # other hand, it now works.
            #os.kill(self._optimizer_process.pid, signal.SIGINT)

        elif status == "pausing":
            self._experiment.add_pausing(candidate)
        elif status == "working":
            self._experiment.add_working(candidate)

    def get_best_candidate(self):
        """
        Returns the best candidate to date.

        Returns
        -------
        best_candidate : candidate or None
            Returns a candidate if there is a best one (which corresponds to
            at least one candidate evaluated) or None if none exists.
        """
        return self._experiment.best_candidate

    def _create_experiment_directory(self):
        global_start_date = time.time()
        date_name = datetime.datetime.utcfromtimestamp(
                global_start_date).strftime("%Y-%m-%d_%H:%M:%S")
        self._experiment_directory_base = os.path.join(self._write_directory_base,
                                    self._experiment.name + "_" + date_name)
        ensure_directory_exists(self._experiment_directory_base)

    def _best_result_per_step_dicts(self, color="b", plot_up_to=None):
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
        x, step_eval, step_best = self._best_result_per_step_data(plot_up_to=plot_up_to)

        step_eval_dict = {
            "x": x,
            "y": step_eval,
            "type": "scatter",
            "label": "%s, current result" %(str(self._experiment.name)),
            "color": color
        }

        step_best_dict = {
            "x": x,
            "y": step_best,
            "type": "line",
            "color": color,
            "label": "%s, best result" %(str(self._experiment.name))
        }

        #print [step_eval_dict, step_best_dict]

        return [step_eval_dict, step_best_dict]

    def _best_result_per_step_data(self, plot_up_to=None):
        """
        This internal function returns goodness of the results by step.
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
        x = []
        step_evaluation = []
        step_best = []
        best_candidate = None
        if plot_up_to is None:
            plot_up_to = len(self._experiment.candidates_finished)
        for i, e in enumerate(self._experiment.candidates_finished[:plot_up_to]):
            x.append(i)
            step_evaluation.append(e.result)
            if self._experiment.better_cand(e, best_candidate):
                best_candidate = e
                step_best.append(e.result)

            else:
                step_best.append(best_candidate.result)
        return x, step_evaluation, step_best

    def _kill_optimizer(self):
        """
        INTERNAL USE ONLY

        Kills the currently running optimizer process (if available) and closes
        the queue that optimizer is pushing to.

        The first is done by sending the SIGINT signal to the process (which
        has been defined as telling the optimizer it is supposed to start
        cleanup operations). This is necessary because we cannot be sure that
        the optimizer isn't currently refitting (in which case an event-based
        solution would mean we'd be needlessly computing), and because it's
        irresponsible to just terminate the process when we can't know whether
        it has created sub-processes.

        Closing the queue is necessary since, after killing the optimizer,
        we can't be sure of its integrity and because we don't need any of the
        old candidates anyways.
        """
        if self._optimizer_in_queue is not None:
            self._optimizer_in_queue.put("exit")
            #TODO Commenting out the below means we cannot kill the optimizer
            # during optimization, even when we have new information. On the
            # other hand, it now works.
            #os.kill(self._optimizer_process.pid, signal.SIGINT)
        if self._optimizer_in_queue is not None:
            self._optimizer_in_queue.close()
        if self._optimizer_queue is not None:
            self._optimizer_queue.close()


    def _build_new_optimizer(self):
        """
        INTERNAL USE ONLY.

        This function builds a new optimizer.

        In general, it instantiates a new optimizer_queue, a new optimizer
        (with the current experiment) and starts said optimizer. It also
        ensures that the old optimizer has been properly killed before.
        """
        self._kill_optimizer()

        self._optimizer_queue = multiprocessing.Queue()
        self._optimizer_in_queue = multiprocessing.Queue()
        self._optimizer_process = check_optimizer(self._optimizer, self._experiment,
                            self._optimizer_queue, self._optimizer_in_queue,
                            optimizer_arguments=self._optimizer_arguments)
        self._optimizer_process.start()


    def _append_to_detailed_csv(self):
        if len(self._experiment.candidates_finished) <= self._csv_steps_written:
            return

        #create file and header if
        wHeader = False
        if self._csv_steps_written == 0:
            #set use header
            wHeader = True

        csv_string, steps_included = self._experiment.to_csv_results(wHeader=wHeader,
                                            fromIndex=self._csv_steps_written)

        #write
        filename = os.path.join(self._experiment_directory_base,
                                self._experiment.name + "_results.csv")

        with open(filename, 'a+') as detailed_file:
            detailed_file.write(csv_string)

        self._csv_steps_written += steps_included

    def get_candidates(self):
        result = {}
        result["finished"] = self._experiment.candidates_finished
        result["pending"] = self._experiment.candidates_pending
        result["working"] = self._experiment.candidates_working
        return result


    def plot_result_per_step(self, ax=None, color="b",
                             plot_min=None, plot_max=None):
        """
        Returns (and plots) the plt.figure plotting the results over the steps.
        Parameters
        ----------
        show_plot : bool, optional
            Whether to show the plot after creation.
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


        plots = self._best_result_per_step_dicts(color)
        if self._experiment.minimization_problem:
            legend_loc = 'upper right'
        else:
            legend_loc = 'upper left'

        plot_options = {
            "legend_loc": legend_loc,
            "x_label": "steps",
            "y_label": "result",
            "title": "Plot of %s result over the steps."
                     %(str(self._experiment.name))
        }
        fig, ax = plot_lists(plots, ax=ax, fig_options=plot_options, plot_min=plot_min, plot_max=plot_max)

        return fig

    def write_plots(self):
        fig = self.plot_result_per_step()
        filename = "result_per_step_%i" %len(self._experiment.candidates_finished)
        path = self._experiment_directory_base + "/plots"
        ensure_directory_exists(path)
        write_plot_to_file(fig, filename, path)
        plt.close(fig)