__author__ = 'Frederik Diehl'

from apsis.models.experiment import Experiment
from abc import ABCMeta, abstractmethod
import Queue
import sys
from time import sleep
import signal
import multiprocessing


class Optimizer(object):
    """
    This defines a basic optimizer interface.
    """
    __metaclass__ = ABCMeta

    SUPPORTED_PARAM_TYPES = []

    @abstractmethod
    def __init__(self, optimizer_params):
        """
        Initializes the optimizer with the arguments under optimizer_params.

        Parameters
        ----------
        optimizer_arguments : dict
            The parameters for the optimization. Depending on the optimizer,
            different arguments are needed.
        """
        pass

    @abstractmethod
    def get_next_candidates(self, experiment):
        """
        Returns several Candidate objects given an experiment.

        It is the free choice of the optimizer how many Candidates to provide,
        but it will provide at least one.
        Parameters
        ----------
        experiment : Experiment
            The experiment to form the base of the next candidate.

        Returns
        -------
        next_candidate : list of Candidate
            The Candidate to next evaluate.
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


class QueueOptimizer(multiprocessing.Process):
    """
    This defines a basic optimizer interface for server/client architecture.

    The main difference for the developer is the usage of gen_candidates,
    which each optimizer has to implement.

    In general, a QueueOptimizer possesses an out_queue, which it keeps filled
    with candidates up to a specified size. To kill such a QueueOptimizer, the
    experiment_helper kills it with a SIGINT signal (which triggers the
    cleanup method). To use new examples, the QueueOptimizer has to be killed
    and newly initiated.

    Attributes
    ----------
    SUPPORTED_PARAM_TYPES : List of ParamDefs
        Which parameter types this optimizer supports. This must be set by
        each optimizer class, and is used to test whether an experiment is
        supported.
    out_queue : multiprocessing.Queue
        The queue used to communicate with the experiment_helper. This will
        always be attempted to keep full with at least min_candidates
        candidates. However, it cannot be guaranteed it won't be empty.
    min_candidates : int
        The minimum number of candidates out_queue should contain at any point.
    experiment : experiment
        The experiment used to generate new candidates.
    """
    __metaclass__ = ABCMeta

    SUPPORTED_PARAM_TYPES = []
    out_queue = None
    min_candidates = None
    experiment = None

    @abstractmethod
    def __init__(self, optimizer_params, experiment, out_queue,
                 min_candidates=1):
        """
        Initializes the optimizer with the arguments under optimizer_params.

        Parameters
        ----------
        optimizer_arguments : dict
            The parameters for the optimization. Depending on the optimizer,
            different arguments are needed.
        experiment : experiment
            The experiment used to generate new candidates.
        out_queue : multiprocessing.Queue
            The queue used to communicate with the experiment_helper. This will
            always be attempted to keep full with at least min_candidates
            candidates. However, it cannot be guaranteed it won't be empty.
        min_candidates : int
            The minimum number of candidates out_queue should contain at any
            point. Note that no matter this value, candidates will be
            appended whenever the queue is empty.
        """
        self.out_queue = out_queue
        self.min_candidates = min_candidates
        signal.signal(signal.SIGINT, self.terminate_gracefully)
        super(QueueOptimizer, self).__init__()

    def run(self):
        """
        Runs the QueueOptimizer.

        The inner working is such that, once per second, the out_queue is
        checked on whether it is empty or contains less than min_candidates
        candidates. If so, new candidates are generated and appended.
        """
        while True:
            if not self.out_queue.full():
                try:
                    if self.out_queue.empty() or \
                                    self.out_queue.qsize < self.min_candidates:
                        new_candidates = self.gen_candidates()
                        [self.out_queue.put(x, block=False) for x in new_candidates]
                except Queue.Full:
                    pass
                sleep(1)


    def terminate_gracefully(self, _signo, _stack_frame):
        """
        This method allows a QueueOptimizer to exit gracefully.

        It can be reimplemented to terminate subprocesses and similar stuff.
        """
        # Raises SystemExit(0):
        sys.exit(0)

    @abstractmethod
    def gen_candidates(self):
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