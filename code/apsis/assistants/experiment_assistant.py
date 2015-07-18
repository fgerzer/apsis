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

class ExperimentAssistant(object, multiprocessing.Process):
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

    Note that, when programming new functions running in the same process as
    this one (and extensions to this class) several things should be kept in
    mind:
    - It is necessary to send all messages through the send_msg function. This
    enables easier changes later on and - more importantly - avoids a problem
    with sending Connections via Connections (namely, it doesn't work).
    - If a method is provided both internally (method_name) and externally
    (_method_name), internal services must always use the internal method, and
    external ones the external method. Otherwise, internal requests will possibly
    hang indefinitely, and external ones will access a non-synchronized status.

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

    AVAILABLE_STATUS = ["finished", "pausing", "working"]

    optimizer = None
    optimizer_arguments = None
    experiment = None

    _rcv_queue = None

    _optimizer_queue = None
    _optimizer_process = None

    write_directory_base = None
    csv_write_frequency = None
    csv_steps_written = 0
    experiment_directory_base = None

    logger = None

    def __init__(self, name, optimizer, param_defs, experiment=None,
                 optimizer_arguments=None, minimization=True,
                 write_directory_base="/tmp/APSIS_WRITING",
                 experiment_directory_base=None, csv_write_frequency=1):
        self._rcv_queue = multiprocessing.Queue()
        self.logger = get_logger(self)
        self.logger.info("Initializing experiment assistant.")
        self.optimizer = optimizer
        self.optimizer_arguments = optimizer_arguments
        if experiment is None:
            experiment = Experiment(name, param_defs, minimization)
        self.experiment = experiment
        self.csv_write_frequency = csv_write_frequency
        if self.csv_write_frequency != 0:
            self.write_directory_base = write_directory_base
            if experiment_directory_base is not None:
                self.experiment_directory_base = experiment_directory_base
                ensure_directory_exists(self.experiment_directory_base)
            else:
                self._create_experiment_directory()
        self._build_new_optimizer()
        multiprocessing.Process.__init__(self)
        self.logger.info("Experiment assistant for %s successfully "
                         "initialized." %name)


    def run(self):
        """
        The run method, used for receiving and parsing messages.

        This method runs forever, waiting for self._rcv_queue to contain a
        message. If this is the case, it calls the method defined by adding a
        _ to the front of msg["action"] with msg as parameter.
        For more details, see the documentation of this class.

        It will automatically unreduce the connection received via
        "return_pipe".

        If it receives a message whose "action" is "exit", it will stop looking
        for further actions, kill the optimizer, and stop.
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


    def send_msg(self, msg):
        """
        This method is used to send an arbitrary message to the process.

        Care should be taken that msg is a message in an acceptable format.
        This means a dict with string key, containing at least an "action" key
        with a string corresponding to the function that should be called,
        all parameters necessary for said function, and - if a return is
        necessary - a connection object for the key "return_pipe".

        Parameters
        ----------
            msg : dict
                The message to be sent. Must include the following keys:
                "action" : string
                    String corresponding to the function to be called. If the
                    string is "function", the function called will be
                    _function.
                "return_pipe" : Connection, optional
                    A connection object into which the result will be put. It
                    will be reduced (so that we are able to send it over a
                    connection).
                One entry per parameter of the function.
                All other entries will be ignored.
        """
        if "return_pipe" in msg:
            msg["return_pipe"] = reduction.reduce_connection(msg["return_pipe"])
        self._rcv_queue.put(msg)

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

    def _create_experiment_directory(self):
        global_start_date = time.time()
        date_name = datetime.datetime.utcfromtimestamp(
                global_start_date).strftime("%Y-%m-%d_%H:%M:%S")
        self.experiment_directory_base = os.path.join(self.write_directory_base,
                                    self.experiment.name + "_" + date_name)
        ensure_directory_exists(self.experiment_directory_base)

    def get_all_candidates(self):
        conn_rcv, conn_send = multiprocessing.Pipe(duplex=False)
        msg = {"action": "get_all_candidates",
               "return_pipe": conn_send}
        self.send_msg(msg)
        candidates = conn_rcv.recv()
        conn_rcv.close()
        return candidates

    def _get_all_candidates(self, msg):
        return_msg = {
            "candidates_finished": self.experiment.candidates_finished,
            "candidates_pending": self.experiment.candidates_pending,
            "candidates_working": self.experiment.candidates_working
        }
        msg["return_pipe"].send(return_msg)

    def best_result_per_step_dicts(self, color="b"):
        conn_rcv, conn_send = multiprocessing.Pipe(duplex=False)
        msg = {
            "action": "best_result_per_step_dicts",
            "color": color,
            "return_pipe": conn_send
        }
        self.send_msg(msg)
        result = conn_rcv.recv()
        conn_rcv.close()
        return result

    def _best_result_per_step_dicts(self, msg):
        x, step_eval, step_best = self._best_result_per_step_data()
        step_eval_dict = {
            "x": x,
            "y": step_eval,
            "type": "scatter",
            "label": "%s, current result" %(str(self.experiment.name)),
            "color": msg["color"]
        }

        step_best_dict = {
            "x": x,
            "y": step_best,
            "type": "line",
            "color": msg["color"],
            "label": "%s, best result" %(str(self.experiment.name))
        }

        msg["return_pipe"].send([step_eval_dict, step_best_dict])

    def exit(self):
        """
        Exits this experiment assistant's process.
        """
        msg = {"action": "exit"}
        self.send_msg(msg)

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
        (with the current experiment) and starts said optimizer. It also
        ensures that the old optimizer has been properly killed before.
        """
        self._kill_optimizer()

        self._optimizer_queue = multiprocessing.Queue()
        self._optimizer_process = build_queue_optimizer(self.optimizer, self.experiment,
                            self._optimizer_queue,
                            optimizer_arguments=self.optimizer_arguments)
        self._optimizer_process.start()


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