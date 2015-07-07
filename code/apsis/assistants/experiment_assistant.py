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
import signal
import multiprocessing
from multiprocessing import reduction

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
    This ExperimentAssistant is used to parallelize the experiment execution.

    In general, the experiment assistant allows external processes to access
    several functions which are internally abstracted with a queue communication
    structure. This means that this is thread safe, and can be used by more than
    one thread or process at once.

    In general, this class manages one QueueBasedOptimizer process, which can
    communicate with it using the optimizer_queue Queue. In normal operation,
    QueueBasedOptimizer keeps the optimizer_queue filled, and experiment_assistant
    uses it to provide new candidates to evaluate. Once a new candidate has been
    evaluated (that is, returned with the status "finished"), the optimizer
    process is killed with a SIGINT signal (which allows it to terminate
    gracefully) and a new optimizer process is started.

    Communication with the outside happens via several available functions,
    which all work with the same scheme.
    This class instance keeps a _rcv_queue, on which the methods called from the
    outside push messages. Each message is a dict and contains at least an
    "action" field clarifying the action, plus additional fields depending on
    the action. If a reply is needed, the "return_pipe" key has a
    multiprocessing.Pipe object, onto which the answer is pushed.

    Therefore, for each action Action, this class implements two functions.
    Action(args) is called from the outside, initializes the message from the
    args and - if necessary - adds a return_pipe. If a reply is needed, it
    will wait for an answer to be pushed on its end of the return_pipe.
    _Action(msg) is called from the inside, from run (where the conversion from
    msg's "action" field to function name is done automatically; so please use
    the scheme indicated above) and receives the information in msg. If a reply
    is needed (which is function-dependant, not message-dependant) it must answer
    on the return_pipe or risk the calling process to remain stuck.


    Attributes
    ----------
    _rcv_queue : multiprocessing.queue
        The queue used to communicate internally.

    _optimizer_queue : multiprocessing.queue
        The queue the optimizer appends its new candidates.

    _optimizer_process : multiprocessing.Process
        The process of the optimizer.

    """
    _rcv_queue = None

    _optimizer_queue = None
    _optimizer_process = None

    def __init__(self, name, optimizer, param_defs,
                 experiment=None, optimizer_arguments=None, minimization=True,
                 write_directory_base="/tmp/APSIS_WRITING",
                 experiment_directory_base=None, csv_write_frequency=1):
        """
        Initializes the ParallelExperimentAssistant.

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
        self._rcv_queue = multiprocessing.Queue()

        super(ParallelExperimentAssistant, self).\
            __init__(name, optimizer, param_defs, experiment=experiment,
                     optimizer_arguments=optimizer_arguments, minimization=minimization,
                     write_directory_base=write_directory_base, experiment_directory_base=experiment_directory_base,
                     csv_write_frequency=csv_write_frequency)
        self._build_new_optimizer()
        multiprocessing.Process.__init__(self)

    def run(self):
        """
        The run method, used for receiving and parsing messages.

        This method runs forever, waiting for self._rcv_queue to contain a
        message. If this is the case, it calls the method defined by adding a
        _ to the front of msg["action"] with msg as parameter.
        For more details, see the documentation of this class.
        """
        exited = False
        while not exited:
            msg = self._rcv_queue.get(block=True)
            if "return_pipe" in msg:
                msg["return_pipe"] = msg["return_pipe"][0](*msg["return_pipe"][1])
            if msg.get("action", None) == "exit":
                self._kill_optimizer()
                exited = True
            else:
                try:
                    getattr(self, "_" + msg["action"])(msg)
                except:
                    pass


    def get_next_candidate(self):
        """
        Returns the Candidate next to evaluate.

        Returns
        -------
        next_candidate : Candidate or None
            The Candidate object that should be evaluated next. May be None,
            which is to be interpreted as no candidate currently being
            available.
        """
        conn_rcv, conn_send = multiprocessing.Pipe(duplex=False)
        msg = {"action": "get_next_candidate",
               "return_pipe": conn_send}
        self.send_msg(msg)
        next_candidate = conn_rcv.recv()
        conn_rcv.close()
        return next_candidate

    def _get_next_candidate(self, msg):
        """
        INTERNAL USE ONLY

        This method is called when a message with the action
        "get_next_candidate" is received.
        If pending candidates exists for the experiment (pending candidates
        are candidates which have been paused in execution), it will return
        one of them (the first one, though no guarantee will be made that this
        remains that way in the future). Otherwise, it will try to return a
        candidate from the _optimizer_queue. If this is empty, it will return
        a None value.

        Parameters
        ----------
            msg : dict
                The message sent to the process. Includes the following keys:
                "action" : string
                    This will be "get_next_candidate".
                "return_pipe" : multiprocessing.Pipe end point
                    This is a pipe end point into which one can put an object.
                    This is used to return the next candidate.
                    One object will be sent into it, which is either the next
                    Candidate or None, if no candidate is currently available.
                All other keys will be ignored.
        """
        self.logger.info("Returning next candidate.")
        next_candidate = None
        while next_candidate is None:
            if self.experiment.candidates_pending:
                next_candidate = self.experiment.candidates_pending.pop()
            else:
                try:
                    next_candidate = self._optimizer_queue.get()
                except:
                    next_candidate = None
        msg["return_pipe"].send(next_candidate)

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
        msg = {
            "action": "update",
            "candidate": candidate,
            "status": status
        }
        self.send_msg(msg)

    def _update(self, msg):
        """
        INTERNAL USE ONLY

        This method is called when a message with the action "update" is
        received.
        Depending on the status, three different actions will occur.
        - finished: The candidate will be added to the experiments' finished
        list. If necessary, the csv results will be written. The optimizer will
        be killed, and a new one initialized.
        - pausing: The candidate will be added to the pending list. It will be
        resumed before any new candidates will be generated by the optimizer.
        - working: The candidate will now be assumed to be worked on.

        Parameters
        ----------
            msg : dict
                The message sent to the process. Includes the following keys:
                "action" : string
                    This will be "update".
                "candidate": Candidate
                    The candidate whose status is updated.
                "status": string
                    The new status of the candidate. Can be one of "working",
                    "pausing" and "finished".
                All other keys will be ignored.
        """
        status = msg["status"]
        candidate = msg["candidate"]
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

            #invoke the writing to files
            step = len(self.experiment.candidates_finished)
            if self.csv_write_frequency != 0 and step != 0 \
                    and step % self.csv_write_frequency == 0:
                self._append_to_detailed_csv()

            # And we rebuild the new optimizer.
            self._build_new_optimizer()

        elif status == "pausing":
            self.experiment.add_pausing(candidate)
        elif status == "working":
            self.experiment.add_working(candidate)


    def _kill_optimizer(self):
        if self._optimizer_process is not None:
        # This sends SIGINT to the optimizer process, which is used to
            # terminate the process irrespective of its current computation.
            # (But, since it will be catched, it can still do cleanup)
            os.kill(self._optimizer_process.pid, signal.SIGINT)
        if self._optimizer_queue is not None:
            self._optimizer_queue.close()

    def _build_new_optimizer(self):
        """
        INTERNAL USE ONLY.

        This function builds a new optimizer.

        In general, it instantiates a new optimizer_queue, a new optimizer
        (with the current experiment) and starts said optimizer.
        """
        self._kill_optimizer()


        self._optimizer_queue = multiprocessing.Queue()
        self._optimizer_process = build_queue_optimizer(self.optimizer, self.experiment,
                            self._optimizer_queue,
                            optimizer_arguments=self.optimizer_arguments)
        self._optimizer_process.start()

    def get_best_candidate(self):
        """
        Returns the best candidate to date.

        Returns
        -------
        best_candidate : candidate or None
            Returns a candidate if there is a best one (which corresponds to
            at least one candidate evaluated) or None if none exists.
        """
        conn_rcv, conn_send = multiprocessing.Pipe(duplex=False)
        msg = {
            "action": "get_best_candidate",
            "return_pipe": conn_send
        }
        self.send_msg(msg)
        best_candidate = conn_rcv.recv()
        conn_rcv.close()
        return best_candidate

    def _get_best_candidate(self, msg):
        """
        INTERNAL USE ONLY

        This method is called when a message with the action
        "get_best_candidate" is received.
        Unsurprisingly, it returns the best existing candidate (or None if no
        such candidate exists).

        Parameters
        ----------
            msg : dict
                The message sent to the process. Includes the following keys:
                "action" : string
                    This will be "get_best_candidate".
                "return_pipe" : multiprocessing.Pipe end point
                    This is a pipe end point into which one can put an object.
                    This is used to return the best candidate.
                    One object will be sent into it, which is either the next
                    Candidate or None, if no best candidate exists.
                All other keys will be ignored.
        """
        msg["return_pipe"].send(self.experiment.best_candidate)

    def send_msg(self, msg):
        if "return_pipe" in msg:
            msg["return_pipe"] = reduction.reduce_connection(msg["return_pipe"])
        self._rcv_queue.put(msg)

    def exit(self):
        msg = {"action": "exit"}
        self.send_msg(msg)