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

AVAILABLE_STATUS = ["finished", "pausing", "working"]

class ExperimentAssistant(multiprocessing.Process):


    _optimizer = None
    _optimizer_arguments = None
    _experiment = None

    _rcv_queue = None

    _optimizer_queue = None
    _optimizer_process = None

    _write_directory_base = None
    _csv_write_frequency = None
    _csv_steps_written = 0
    _experiment_directory_base = None

    _logger = None

    def __init__(self, rcv_queue, name, optimizer, param_defs, experiment=None,
                 optimizer_arguments=None, minimization=True,
                 write_directory_base="/tmp/APSIS_WRITING",
                 experiment_directory_base=None, csv_write_frequency=1):
        self._logger = get_logger(self)
        self._logger.info("Initializing experiment assistant.")
        self._rcv_queue = rcv_queue
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
        multiprocessing.Process.__init__(self)
        self._logger.info("Experiment assistant for %s successfully "
                         "initialized." %name)

    def run(self):
        exited = False
        try:
            while not exited:
                msg = self._rcv_queue.get(block=True)
                if msg.get("action", None) == "exit":
                    exited = True
                    continue
                if "result_conn" in msg:
                    msg["result_conn"] = msg["result_conn"][0](*msg["result_conn"][1])
                try:
                    getattr(self, "_" + msg["action"])(msg)
                except:
                    pass
        finally:
            self._kill_optimizer()

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
                "result_conn" : multiprocessing.Pipe end point
                    This is a pipe end point into which one can put an object.
                    This is used to return the next candidate.
                    One object will be sent into it, which is either the next
                    Candidate or None, if no candidate is currently available.
                All other keys will be ignored.
        """

        self._logger.info("Returning next candidate.")
        next_candidate = None
        while next_candidate is None:
            if self._experiment.candidates_pending:
                next_candidate = self._experiment.candidates_pending.pop()
            else:
                try:
                    next_candidate = self._optimizer_queue.get()
                except:
                    next_candidate = None
        msg["result_conn"].send(next_candidate)


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

            # And we rebuild the new optimizer.
            self._build_new_optimizer()

        elif status == "pausing":
            self._experiment.add_pausing(candidate)
        elif status == "working":
            self._experiment.add_working(candidate)

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
        msg["result_conn"].send(self._experiment.best_candidate)

    def _get_experiment(self, msg):
        msg["result_conn"].send(self._experiment)

    def _get_all_candidates(self, msg):
        return_msg = {
            "candidates_finished": self._experiment.candidates_finished,
            "candidates_pending": self._experiment.candidates_pending,
            "candidates_working": self._experiment.candidates_working
        }
        msg["result_conn"].send(return_msg)

    def _create_experiment_directory(self):
        global_start_date = time.time()
        date_name = datetime.datetime.utcfromtimestamp(
                global_start_date).strftime("%Y-%m-%d_%H:%M:%S")
        self.experiment_directory_base = os.path.join(self._write_directory_base,
                                    self._experiment.name + "_" + date_name)
        ensure_directory_exists(self.experiment_directory_base)

    def _best_result_per_step_dicts(self, msg):
        x, step_eval, step_best = self._best_result_per_step_data()
        step_eval_dict = {
            "x": x,
            "y": step_eval,
            "type": "scatter",
            "label": "%s, current result" %(str(self._experiment.name)),
            "color": msg["color"]
        }

        step_best_dict = {
            "x": x,
            "y": step_best,
            "type": "line",
            "color": msg["color"],
            "label": "%s, best result" %(str(self._experiment.name))
        }

        msg["result_conn"].send([step_eval_dict, step_best_dict])

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
        for i, e in enumerate(self._experiment.candidates_finished):
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
        self._optimizer_process = build_queue_optimizer(self._optimizer, self._experiment,
                            self._optimizer_queue,
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
        filename = os.path.join(self.experiment_directory_base,
                                self._experiment.name + "_results.csv")

        with open(filename, 'a+') as detailed_file:
            detailed_file.write(csv_string)

        self._csv_steps_written += steps_included