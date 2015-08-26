__author__ = 'Frederik Diehl'

from apsis.models.experiment import Experiment
from abc import ABCMeta, abstractmethod
import Queue
import sys
from time import sleep
import signal
import multiprocessing
import Queue

class Optimizer(object):
    __metaclass__ = ABCMeta

    SUPPORTED_PARAM_TYPES = []

    _experiment = None

    def __init__(self, optimizer_params, experiment):
        self._experiment = experiment

    def update(self, experiment):
        self._experiment = experiment

    @abstractmethod
    def get_next_candidates(self, num_candidates=1):
        pass

    def exit(self):
        pass



    def _is_experiment_supported(self, experiment):
        """
        Tests whether all parameter types in experiment are supported by this
        optimizer.

        Parameters
        ----------
        experiment : Experiment
            The experiment to test.

        Returns
        -------
        supported : bool
            False iff one or more of experiment's parameter definitions are not
            supported.
        """
        for name, pd in experiment.parameter_definitions.iteritems():
            if not self._is_supported_param_type(pd):
                return False
        return True

    def _is_supported_param_type(self, param):
        """
        Tests whether a certain parameter is supported by the optimizer.

        Parameters
        ----------
        param :
            The parameter to be tested

        Result
        ------
        is_supported : bool
            True iff param is supported by this optimizer.
        """
        if isinstance(self.SUPPORTED_PARAM_TYPES, list):
            for sup in self.SUPPORTED_PARAM_TYPES:
                if isinstance(param, sup):
                    return True

        return False

class QueueBasedOptimizer(Optimizer):
    _optimizer_in_queue = None
    _optimizer_out_queue = None

    _optimizer_process = None

    def __init__(self, optimizer_class, optimizer_params, experiment):
        self._optimizer_in_queue = multiprocessing.Queue()
        self._optimizer_out_queue = multiprocessing.Queue()

        optimizer = QueueBackend(optimizer_class, optimizer_params, experiment,
                                 self._optimizer_out_queue, self._optimizer_in_queue)
        optimizer.start()

        super(QueueBasedOptimizer, self).__init__(optimizer_params,
                                                  experiment)

    def get_next_candidates(self, num_candidates=1):
        next_candidates = []
        try:
            for i in range(num_candidates):
                new_candidate = self._optimizer_out_queue.get_nowait()
                next_candidates.append(new_candidate)
        except Queue.Empty:
            pass
        return next_candidates


    def update(self, experiment):
        self._optimizer_in_queue.put(experiment)

    def exit(self):
        if self._optimizer_in_queue is not None:
            self._optimizer_in_queue.put("exit")
        if self._optimizer_in_queue is not None:
            self._optimizer_in_queue.close()
        if self._optimizer_out_queue is not None:
            self._optimizer_out_queue.close()

class QueueBackend(multiprocessing.Process):
    _experiment = None
    _out_queue = None
    _in_queue = None

    _optimizer = None

    _min_candidates = None
    _exited = None

    def __init__(self, optimizer_class, optimizer_params, experiment, out_queue, in_queue):

        self._out_queue = out_queue
        self._in_queue = in_queue
        self._min_candidates = optimizer_params.get("min_candidates", 5)
        self._optimizer = optimizer_class(optimizer_params, experiment)
        self._exited = False
        self._experiment = experiment
        multiprocessing.Process.__init__(self)

    def run(self):
        try:
            while not self._exited:
                self._check_generation()
                self._check_update()
                sleep(0.1)
        finally:
            if self._in_queue is not None:
                self._in_queue.close()
            if self._out_queue is not None:
                self._out_queue.close()

    def _check_update(self):
        new_update = None
        while not self._in_queue.empty():
            try:
                new_update = self._in_queue.get_nowait()
            except Queue.Empty:
                pass
            if new_update == "exit":
                self._exited = True
                return
        if new_update is not None:
            # clear the out queue. We'll soon have new information.
            try:
                while not self._out_queue.empty():
                    self._out_queue.get_nowait()
            except Queue.Empty:
                pass
            self._experiment = new_update
            self._optimizer.update(self._experiment)

    def _check_generation(self):
        try:
            if (self._out_queue.empty() or
                           self._out_queue.qsize < self._min_candidates):
                new_candidates = self._optimizer.get_next_candidates(num_candidates=self._min_candidates)
                if new_candidates is None:
                    return
                for c in new_candidates:
                    self._out_queue.put_nowait(c)
        except Queue.Full:
            return