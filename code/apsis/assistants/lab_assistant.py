#TODO write plotting functions.

__author__ = 'Frederik Diehl'

from apsis.assistants.experiment_assistant import ExperimentAssistant
import matplotlib.pyplot as plt
from apsis.utilities.plot_utils import create_figure, _polish_figure, plot_lists, write_plot_to_file
from apsis.utilities.file_utils import ensure_directory_exists
import time
import datetime
import os
from apsis.utilities.logging_utils import get_logger
import numpy as np
import multiprocessing
from multiprocessing import reduction
import sys

class LabAssistant(multiprocessing.Process):
    """
    This LabAssistant is used to parallelize the experiment execution.

    In general, the lab assistant allows external processes to access
    several functions which are internally abstracted with a queue communication
    structure. This means that this is thread safe, and can be used by more than
    one thread or process at once.

    In general, this class manages several ParallelExperimentAssistant
    processes, with which it communicates via their methods.

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
    """

    _rcv_queue = None
    _exp_assistants = None

    write_directory_base = None
    lab_run_directory = None
    global_start_date = None
    logger = None

    def __init__(self, write_directory_base="/tmp/APSIS_WRITING"):
        """
        Initializes the LabAssistant.

        This is identical to the PrettyLabAssistant's init method (and calls
        it) except for also initializing _rcv_queue and (hardcoded)
        initializing Process.
        """
        self.logger = get_logger(self)
        self.logger.info("Initializing laboratory assistant.")

        self._rcv_queue = multiprocessing.Queue()
        self._exp_assistants = {}

        self.write_directory_base = write_directory_base
        self.global_start_date = time.time()

        self._init_directory_structure()
        self.logger.info("laboratory assistant successfully initialized.")
        multiprocessing.Process.__init__(self)


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

            ensure_directory_exists(self.lab_run_directory)

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
        try:
            while not exited:
                msg = self._rcv_queue.get(block=True)
                print("\tLAss run received msg %s" %msg)
                if "return_pipe" in msg:
                    msg["return_pipe"] = msg["return_pipe"][0](*msg["return_pipe"][1])
                    print("\tLAss unpacked msg %s" %msg)
                if msg.get("action", None) == "exit":
                    exited = True
                else:
                    try:
                        getattr(self, "_" + msg["action"])(msg)
                    except:
                        pass
        finally:
            for exp_ass in self._exp_assistants.values():
                exp_ass.exit()


    def init_experiment(self, name, optimizer, param_defs,
                        optimizer_arguments=None, minimization=True):
        """
        Initializes a new experiment.

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
        msg = {
            "action": "init_experiment",
            "name": name,
            "optimizer": optimizer,
            "param_defs": param_defs,
            "optimizer_arguments": optimizer_arguments,
            "minimization": minimization
        }
        self.send_msg(msg)


    def _init_experiment(self, msg):
        """
        INTERNAL USE ONLY

        Initializes a new experiment.

        This is done by checking whether the proposed name is in the available
        names, and if not, initializing and starting it.

        Parameters
        ----------
        msg : dict
            The message encoding the parameters. Includes the following keys:
            "action" : string
                This will be "init_experiment"
            "optimizer" : string
                This encodes the optimizer.
            "param_defs" : list
                This is a list of parameter definitions for the experiment.
            "optimizer_arguments" : dict
                A dict of optimizer parameters.
            "minimization" : bool
                Whether this experiment's goal is to minimize or maximize the
                candidate results.
            All other parameter will be ignored.
        """
        name = msg["name"]
        optimizer = msg["optimizer"]
        param_defs = msg["param_defs"]
        optimizer_arguments = msg["optimizer_arguments"]
        minimization = msg["minimization"]
        if name in self._exp_assistants:
            raise ValueError("Already an experiment with name %s registered."
                             %name)
        self._exp_assistants[name] = ExperimentAssistant(name, optimizer,
                                param_defs, optimizer_arguments=optimizer_arguments,
                                minimization=minimization,
                                write_directory_base=self.lab_run_directory,
                                csv_write_frequency=1)
        self._exp_assistants[name].start()
        self.logger.info("Experiment initialized successfully.")

    def get_next_candidate(self, exp_name, conn_send=None):
        """
        Returns the Candidate next to evaluate for a specific experiment.

        Parameters
        ----------
        exp_name : string
            Has to be in experiment_assistants.

        Returns
        -------
        next_candidate : Candidate or None:
            The Candidate object that should be evaluated next. May be None.
        """
        return_value = False
        if conn_send is None:
            return_value = True
            conn_rcv, conn_send = multiprocessing.Pipe(duplex=False)
        msg = {"action": "get_next_candidate",
                   "exp_name": exp_name,
                   "return_pipe": conn_send}

        self.send_msg(msg)
        if return_value:
            next_candidate = conn_rcv.recv()
            conn_rcv.close()
            return next_candidate

    def _get_next_candidate(self, msg):
        """
        INTERNAL USE ONLY

        This simply multiplexes the function to the corresponding
        exp_assistant.
        """
        exp_assistant_name = msg["exp_name"]
        conn_send = msg["return_pipe"]
        self._exp_assistants[exp_assistant_name].get_next_candidate(conn_send)

    def update(self, exp_name, candidate, status="finished"):
        """
        Updates the experiment with the status of an experiment
        evaluation.

        Parameters
        ----------
        exp_name : string
            Has to be in experiment_assistants
        candidate : Candidate
            The Candidate object whose status is updated.
        status : {"finished", "pausing", "working"}, optional
            A string defining the status change. Can be one of the following:
            - finished: The Candidate is now finished.
            - pausing: The evaluation of Candidate has been paused and can be
                resumed by another worker.
            - working: The Candidate is now being worked on by a worker.
        """
        msg = {"action": "update",
                   "exp_name": exp_name,
                   "status": status,
                   "candidate": candidate}
        self.send_msg(msg)

    def _update(self, msg):
        """
        INTERNAL USE ONLY

        This simply multiplexes the function to the corresponding
        exp_assistant.
        """
        self._exp_assistants[msg["exp_name"]].update(candidate=msg["candidate"],
                                                    status=msg["status"])


    def get_best_candidate(self, exp_name, conn_send=None):
        """
        Returns the best candidate to date for a specific experiment.

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
        return_value = False
        if conn_send is None:
            return_value = True
            conn_rcv, conn_send = multiprocessing.Pipe(duplex=False)
        msg = {
            "action": "get_best_candidate",
            "exp_name": exp_name,
            "return_pipe": conn_send
        }
        self.send_msg(msg)
        if return_value:
            best_candidate = conn_rcv.recv()
            conn_rcv.close()
            return best_candidate

    def _get_best_candidate(self, msg):
        """
        INTERNAL USE ONLY

        This simply multiplexes the function to the corresponding
        exp_assistant.
        """
        best_candidate = self._exp_assistants[msg["exp_name"]].get_best_candidate(msg["return_pipe"])

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

    def exit(self):
        """
        Exits this experiment assistant's process.
        """
        msg = {"action": "exit"}
        self.send_msg(msg)

    def get_all_candidates(self, exp_name, conn_send):
        return_value = False
        if conn_send is None:
            return_value = True
            conn_rcv, conn_send = multiprocessing.Pipe(duplex=False)
        msg = {
            "action": "get_all_candidates",
            "exp_name": exp_name,
            "return_pipe": conn_send
        }
        self.send_msg(msg)
        if return_value:
            all_candidates = conn_rcv.recv()
            conn_rcv.close()
            return all_candidates

    def _get_all_candidates(self, msg):
        all_candidates = self._exp_assistants[msg["exp_name"]].get_all_candidates(msg["return_pipe"])

    def get_all_experiments(self, conn_send=None):
        return_value = False
        if conn_send is None:
            return_value = True
            conn_rcv, conn_send = multiprocessing.Pipe(duplex=False)
        msg = {
            "action": "get_all_experiments",
            "return_pipe": conn_send
        }
        self.send_msg(msg)
        if return_value:
            all_experiments = conn_rcv.recv()
            conn_rcv.close()
            return all_experiments

    def _get_all_experiments(self, msg):
        send_conn = msg["return_pipe"]
        all_exps = {}
        print("bla 0")
        for exp_ass_name, exp_ass in self._exp_assistants:
            print(exp_ass_name)
            conn_rcv, conn_send = multiprocessing.Pipe(duplex=False)
            exp_ass.get_experiment(conn_rcv)
            print("bla 0.5")
            exp = conn_rcv.recv()
            all_exps[exp_ass_name] = exp
        print("bla 1")
        send_conn.send(all_exps)
        print("bla2")