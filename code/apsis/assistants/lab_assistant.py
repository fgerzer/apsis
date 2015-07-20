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

timeout_answer = 1

ACTIONS_FOR_EXP = ["get_next_candidate", "update", "get_best_candidate",
                   "get_all_candidates"]

class LabAssistant(multiprocessing.Process):
    _rcv_queue = None
    _exp_ass_queues = None

    _write_directory_base = None
    _lab_run_directory = None
    _global_start_date = None
    _logger = None



    def __init__(self, rcv_queue, write_directory_base="/tmp/APSIS_WRITING"):
        #TODO add check for windows for directory name in here, via os.name=="nt"
        self._logger = get_logger(self)
        self._logger.info("Initializing lab assistant.")
        self._rcv_queue = rcv_queue
        self._exp_ass_queues = {}
        self._write_directory_base = write_directory_base
        self._global_start_date = time.time()
        self._init_directory_structure() #TODO
        multiprocessing.Process.__init__(self)
        self._logger.info("lab assistant successfully initialized.")

    def run(self):
        exited = False
        try:
            while not exited:
                msg = self._rcv_queue.get(block=True)
                if msg.get("action", None) == "exit":
                    exited = True
                    continue
                if "exp_name" in msg and msg.get("action", None) in ACTIONS_FOR_EXP:
                    self._exp_ass_queues[msg["exp_name"]].put(msg)
                else:
                    try:
                        getattr(self, "_" + msg["action"])(msg)
                    except Exception as e:
                        print("EXCEPTION in lab ass: %s" %e)
                        pass
                    finally:
                        pass
        finally:
            for exp_name in self._exp_ass_queues:
                msg = {"action": "exit"}
                self._exp_ass_queues[exp_name].put(msg)

    def _init_experiment(self, msg):
        name = msg["name"]
        if name in self._exp_ass_queues:
            raise ValueError("Already an experiment with name %s registered"
                             %name)
        optimizer = msg["optimizer"]
        param_defs = msg["param_defs"]
        optimizer_arguments = msg["optimizer_arguments"]
        minimization = msg["minimization"]
        rcv_queue_exp_ass = multiprocessing.Queue()

        self._exp_ass_queues[name] = rcv_queue_exp_ass
        exp_ass = ExperimentAssistant(rcv_queue_exp_ass, name, optimizer,
                            param_defs, optimizer_arguments=optimizer_arguments,
                            minimization=minimization,
                            write_directory_base=self._lab_run_directory,
                            csv_write_frequency=1)
        exp_ass.start()
        self._logger.info("Experiment initialized successfully.")

    def _get_all_experiments(self, msg):
        all_exps = {}
        for exp_ass_name, exp_ass_queue in self._exp_ass_queues:
            conn_rcv, conn_send = multiprocessing.Pipe(duplex=False)
            msg_exp = {
                "action": "get_experiment",
                "result_conn": conn_send
            }
            self._send_msg_exp(exp_ass_name, msg_exp)
            exp = conn_rcv.recv()
            all_exps[exp_ass_name] = exp
        msg["result_queue"].put(all_exps)

    def _send_msg_exp(self, exp_name, msg):
        msg["result_conn"] = reduction.reduce_connection(msg["result_conn"])
        self._exp_ass_queues[exp_name].send(msg)


    def _init_directory_structure(self):
        """
        Method to create the directory structure if not exists
        for results and plots writing
        """
        if self._lab_run_directory is None:
            date_name = datetime.datetime.utcfromtimestamp(
                self._global_start_date).strftime("%Y-%m-%d_%H:%M:%S")

            self._lab_run_directory = os.path.join(self._write_directory_base,
                                                  date_name)

            ensure_directory_exists(self._lab_run_directory)