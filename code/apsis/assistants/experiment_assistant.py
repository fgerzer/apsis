__author__ = 'Frederik Diehl'

from apsis.models.experiment import Experiment
from apsis.models.candidate import Candidate
from apsis.utilities.optimizer_utils import check_optimizer, build_queue_optimizer
from apsis.utilities.file_utils import ensure_directory_exists
import matplotlib.pyplot as plt
from apsis.utilities.plot_utils import plot_lists, create_figure
import datetime
import os
import time
from apsis.utilities.logging_utils import get_logger
import Queue
import sys

import multiprocessing
import signal

class BasicExperimentAssistant(object):
    """
    This ExperimentAssistant assists with executing experiments.

    It provides methods for getting candidates to evaluate, returning the
    evaluated Candidate and administrates the optimizer.

    Attributes
    ----------
    optimizer : Optimizer
        This is an optimizer implementing the corresponding functions: It
        gets an experiment instance, and returns one or multiple candidates
        which should be evaluated next.
    optimizer_arguments : dict
        These are arguments for the optimizer. Refer to their documentation
        as to which are available.
    experiment : Experiment
        The experiment this assistant assists with.
    write_directory_base : string or None
        The directory to write all results to. If not
        given, a directory with timestamp will automatically be created
        in write_directory_base
    csv_write_frequency : int
        States how often the csv file should be written to.
        If set to 0 no results will be written.
    logger : logging.logger
        The logger for this class.
    """

    AVAILABLE_STATUS = ["finished", "pausing", "working"]

    optimizer = None
    optimizer_arguments = None
    experiment = None

    write_directory_base = None
    experiment_directory_base = None
    csv_write_frequency = None
    csv_steps_written = 0

    logger = None

    def __init__(self, name, optimizer, param_defs, experiment=None, optimizer_arguments=None,
                 minimization=True, write_directory_base="/tmp/APSIS_WRITING",
                 experiment_directory_base=None, csv_write_frequency=1):
        """
        Initializes the BasicExperimentAssistant.

        Parameters
        ----------
        name : string
            The name of the experiment. This does not have to be unique, but is
            for human orientation.
        optimizer : Optimizer instance or string
            This is an optimizer implementing the corresponding functions: It
            gets an experiment instance, and returns one or multiple candidates
            which should be evaluated next.
            Alternatively, it can be a string corresponding to the optimizer,
            as defined by apsis.utilities.optimizer_utils.
        param_defs : dict of ParamDef.
            This is the parameter space defining the experiment.
        experiment : Experiment
            Preinitialize this assistant with an existing experiment.
        optimizer_arguments=None : dict
            These are arguments for the optimizer. Refer to their documentation
            as to which are available.
        minimization=True : bool
            Whether the problem is one of minimization or maximization.
        write_directory_base : string, optional
            The global base directory for all writing. Will only be used
            for creation of experiment_directory_base if this is not given.
        experiment_directory_base : string or None, optional
            The directory to write all the results to. If not
            given a directory with timestamp will automatically be created
            in write_directory_base
        csv_write_frequency : int, optional
            States how often the csv file should be written to.
            If set to 0 no results will be written.
        """
        self.logger = get_logger(self)
        self.logger.info("Initializing experiment assistant.")
        self.optimizer = optimizer
        self.optimizer_arguments = optimizer_arguments

        if experiment is None:
            self.experiment = Experiment(name, param_defs, minimization)
        else:
            self.experiment = experiment

        self.csv_write_frequency = csv_write_frequency

        if self.csv_write_frequency != 0:
            self.write_directory_base = write_directory_base
            if experiment_directory_base is not None:
                self.experiment_directory_base = experiment_directory_base
                ensure_directory_exists(self.experiment_directory_base)
            else:
                self._create_experiment_directory()
        self.logger.info("Experiment assistant successfully initialized.")

    def get_next_candidate(self):
        """
        Returns the Candidate next to evaluate.

        Returns
        -------
        next_candidate : Candidate or None
            The Candidate object that should be evaluated next. May be None.
        """
        self.logger.info("Returning next candidate.")
        self.optimizer = check_optimizer(self.optimizer,
                                optimizer_arguments=self.optimizer_arguments)
        if not self.experiment.candidates_pending:
            self.experiment.candidates_pending.extend(
                self.optimizer.get_next_candidates(self.experiment))
        next_candidate = self.experiment.candidates_pending.pop()
        self.logger.info("next candidate found: %s" %next_candidate)
        return next_candidate



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
        if status not in self.AVAILABLE_STATUS:
            message = ("status not in %s but %s."
                             %(str(self.AVAILABLE_STATUS), str(status)))
            self.logger.error(message)
            raise ValueError(message)

        if not isinstance(candidate, Candidate):
            message = ("candidate %s not a Candidate instance."
                             %str(candidate))
            self.logger.error(message)
            raise ValueError(message)

        self.logger.info("Got new %s of candidate %s with parameters %s"
                         " and result %s" %(status, candidate, candidate.params,
                                            candidate.result))

        if status == "finished":
            self.experiment.add_finished(candidate)
            #Also delete all pending candidates from the experiment - we have
            #new data available.
            self.experiment.candidates_pending = []

            #invoke the writing to files
            step = len(self.experiment.candidates_finished)
            if self.csv_write_frequency != 0 and step != 0 \
                    and step % self.csv_write_frequency == 0:
                self._append_to_detailed_csv()

        elif status == "pausing":
            self.experiment.add_pausing(candidate)
        elif status == "working":
            self.experiment.add_working(candidate)

    def get_best_candidate(self):
        """
        Returns the best candidate to date.

        Returns
        -------
        best_candidate : candidate or None
            Returns a candidate if there is a best one (which corresponds to
            at least one candidate evaluated) or None if none exists.
        """
        return self.experiment.best_candidate

    def _append_to_detailed_csv(self):
        if len(self.experiment.candidates_finished) <= self.csv_steps_written:
            return

        #create file and header if
        wHeader = False
        if self.csv_steps_written == 0:
            #set use header
            wHeader = True

        csv_string, steps_included = self.experiment.to_csv_results(wHeader=wHeader,
                                            fromIndex=self.csv_steps_written)

        #write
        filename = os.path.join(self.experiment_directory_base,
                                self.experiment.name + "_results.csv")

        with open(filename, 'a+') as detailed_file:
            detailed_file.write(csv_string)

        self.csv_steps_written += steps_included

    def _create_experiment_directory(self):
        global_start_date = time.time()

        date_name = datetime.datetime.utcfromtimestamp(
                global_start_date).strftime("%Y-%m-%d_%H:%M:%S")

        self.experiment_directory_base = os.path.join(self.write_directory_base,
                                    self.experiment.name + "_" + date_name)

        ensure_directory_exists(self.experiment_directory_base)


class PrettyExperimentAssistant(BasicExperimentAssistant):
    """
    A 'prettier' version of the experiment assistant, mostly through plots.
    """

    def plot_result_per_step(self, show_plot=True, ax=None, color="b",
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
        if self.experiment.minimization_problem:
            legend_loc = 'upper right'
        else:
            legend_loc = 'upper left'

        plot_options = {
            "legend_loc": legend_loc,
            "x_label": "steps",
            "y_label": "result",
            "title": "Plot of %s result over the steps."
                     %(str(self.experiment.name))
        }
        fig, ax = plot_lists(plots, ax=ax, fig_options=plot_options, plot_min=plot_min, plot_max=plot_max)

        if show_plot:
            plt.show(True)

        return ax

    def _best_result_per_step_dicts(self, color="b"):
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
        x, step_eval, step_best = self._best_result_per_step_data()

        step_eval_dict = {
            "x": x,
            "y": step_eval,
            "type": "scatter",
            "label": "%s, current result" %(str(self.experiment.name)),
            "color": color
        }

        step_best_dict = {
            "x": x,
            "y": step_best,
            "type": "line",
            "color": color,
            "label": "%s, best result" %(str(self.experiment.name))
        }

        #print [step_eval_dict, step_best_dict]

        return [step_eval_dict, step_best_dict]

    def _best_result_per_step_data(self):
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
        for i, e in enumerate(self.experiment.candidates_finished):
            x.append(i)
            step_evaluation.append(e.result)
            if self.experiment.better_cand(e, best_candidate):
                best_candidate = e
                step_best.append(e.result)

            else:
                step_best.append(best_candidate.result)
        return x, step_evaluation, step_best

class ParallelExperimentAssistant(PrettyExperimentAssistant, multiprocessing.Process):
    """
    This class implements a client/server architecture for the exp_assistant.

    In general, this works as follows: The PEA contains two queues.
    update_queue receives new (candidate, status) results from workers (via the
    lab helper and REST interface). Update_queue also is able to stop this
     assistant by adding a None value to it.
    next_queue communicates with the lab helper/REST interface by keeping it
    filled with new candidates to be evaluated.
    Additionally, it keeps an optimizer_queue for communicating with the
    optimizer. This is (ideally) kept filled with new candidates to be
    evaluated.
    Lastly, candidate_last_time is a dictionary where each candidate of this
    experiment is assigned the time its worker last sent a status update. This
    is used to find out when a worker (probably) has crashed.
    """
    update_queue = None
    next_queue = None

    optimizer_queue = None
    optimizer_process = None

    # We'll leave this for now, as the first version should just support
    # parallelism in general.
    #candidate_last_time = None
    #max_time_without_update = None

    def __init__(self, name, optimizer, param_defs, update_queue, next_queue,
                 experiment=None, optimizer_arguments=None, minimization=True,
                 write_directory_base="/tmp/APSIS_WRITING",
                 experiment_directory_base=None, csv_write_frequency=1,
                 max_time_without_update=None):

        self.update_queue = update_queue
        self.next_queue = next_queue
        self.candidate_last_time = {}
        #self.max_time_without_update = max_time_without_update
        self._build_new_optimizer()
        super(ParallelExperimentAssistant, self).\
            __init__(name, optimizer, param_defs, experiment=experiment,
                     optimizer_arguments=optimizer_arguments, minimization=minimization,
                     write_directory_base=write_directory_base, experiment_directory_base=experiment_directory_base,
                     csv_write_frequency=csv_write_frequency)

    def run(self):
        while True:
            # if there are new examples, we stop the optimizer and
            # begin a new one.
            if not self.update_queue.empty():
                try:
                    rcv = self.update_queue.get()
                    if rcv is None:
                        # we should stop this assistant.
                        self.optimizer_exit.set()
                        sys.exit(0)
                    candidate, status = rcv
                    self.update(candidate, status)
                except Queue.Empty:
                    pass
            # if we have new possible examples, we add them to the next_queue.
            if not self.optimizer_queue.empty():
                while not self.optimizer_queue.empty():
                    try:
                        self.get_next_candidate()
                    except Queue.Full:
                        pass
            # This, too, is part of the check for stopped candidates.
            #if self.max_time_without_update is not None:
            #    for cand, last_update in self.candidate_last_time:
            #        if time.time() - last_update > self.max_time_without_update:
            #            #candidate has not been heard from for too long.
            time.sleep(1)

    def update(self, candidate, status="finished"):
        if status not in self.AVAILABLE_STATUS:
            message = ("status not in %s but %s."
                             %(str(self.AVAILABLE_STATUS), str(status)))
            self.logger.error(message)
            raise ValueError(message)

        if not isinstance(candidate, Candidate):
            message = ("candidate %s not a Candidate instance."
                             %str(candidate))
            self.logger.error(message)
            raise ValueError(message)

        self.logger.info("Got new %s of candidate %s with parameters %s"
                         " and result %s" %(status, candidate, candidate.params,
                                            candidate.result))

        if status == "finished":
            self.experiment.add_finished(candidate)
            #Also delete all pending candidates from the experiment - we have
            #new data available.
            self.experiment.candidates_pending = []

            #invoke the writing to files
            step = len(self.experiment.candidates_finished)
            if self.csv_write_frequency != 0 and step != 0 \
                    and step % self.csv_write_frequency == 0:
                self._append_to_detailed_csv()
            #and build a new optimizer.
            # This sends SIGINT to the optimizer process, which is used to
            # terminate the process irrespective of its current computation.
            # (But, since it will be catched, it can still do cleanup)
            os.kill(self.optimizer_process.pid, signal.SIGINT)
            self.optimizer_queue.close()

            # And we rebuild the new optimizer.
            self._build_new_optimizer()

        elif status == "pausing":
            self.experiment.add_pausing(candidate)
        elif status == "working":
            self.experiment.add_working(candidate)

    def _build_new_optimizer(self):
        self.optimizer_queue = multiprocessing.Queue()

        self.optimizer = build_queue_optimizer(self.optimizer, self.experiment,
                            self.optimizer_queue,
                            optimizer_arguments=self.optimizer_arguments)
        self.optimizer.start()

    def get_next_candidate(self):
        """
        Returns the Candidate next to evaluate.

        Returns
        -------
        next_candidate : Candidate or None
            The Candidate object that should be evaluated next. May be None.
        """
        self.logger.info("Returning next candidate.")
        if not self.experiment.candidates_pending:
            next_candidate = self.experiment.candidates_pending.pop()
        else:
            if not self.optimizer_queue.empty():
                next_candidate = self.optimizer_queue.get()
            else:
                next_candidate = None
        if next_candidate is not None:
            self.next_queue.append(next_candidate)
            self.logger.info("next candidate found: %s" %next_candidate)
        else:
            self.logger.info("No current candidate available.")