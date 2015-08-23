__author__ = 'Frederik Diehl'

from apsis.models.experiment import Experiment
from abc import ABCMeta, abstractmethod
import Queue
import sys
from time import sleep
import signal
import multiprocessing
import Queue

class Optimizer(multiprocessing.Process):
    __metaclass__ = ABCMeta
    SUPPORTED_PARAM_TYPES = []

    _out_queue = None
    _experiment = None

    _in_queue = None

    _min_candidates = None

    _exited = None

    @abstractmethod
    def __init__(self, optimizer_params, experiment, out_queue, in_queue, min_candidates=5):
        self._out_queue = out_queue
        self._in_queue = in_queue
        self._min_candidates = min_candidates
        #TODO By commenting this out, we risk not inefficiencies. However,
        # due to the inability to catch signals outside the main thread,
        # it is currently necessary.
        #signal.signal(signal.SIGINT, self._update_and_recheck)
        self._exited = False
        multiprocessing.Process.__init__(self)
        self._experiment = experiment

    def run(self):
        """
        Runs the QueueOptimizer.

        The inner working is such that, once per second, the out_queue is
        checked on whether it is empty or contains less than min_candidates
        candidates. If so, new candidates are generated and appended.
        """
        try:
            while not self._exited:
                self._update_and_recheck(None, None)
                if not self._out_queue.full():
                    try:
                        if self._out_queue.empty() or \
                                        self._out_queue.qsize < self._min_candidates:
                            new_candidates = self._gen_candidates(num_candidates=self._min_candidates)
                            [self._out_queue.put(x, block=False) for x in new_candidates]
                    except Queue.Full:
                        pass
                    sleep(0.1)
        finally:
            if self._in_queue is not None:
                self._in_queue.close()
            if self._out_queue is not None:
                self._out_queue.close()

    def _update_and_recheck(self, _signo, _stack_frame):
        new_update = None
        while not self._in_queue.empty():
            try:
                new_update = self._in_queue.get_nowait()
            except Queue.Empty:
                return
            if new_update == "exit":
                self._exited = True
                return
        if new_update is not None:
            self._experiment = new_update
            self._refit()

    @abstractmethod
    def _refit(self):
        pass


    @abstractmethod
    def _gen_candidates(self, num_candidates=1):
        """
        Generates new candidates to add to the out_queue.

        This is the heart of the QueueOptimizer, and the implementation point
         for each new one. Generally, this function should behave as follows:
         On being called, returns a list of candidates next to evaluate.
         Note that, if you don't return enough candidates to bring the queue
         to more than min_candidates, it's probable that this function will be
         called again directly afterwards.
        """
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