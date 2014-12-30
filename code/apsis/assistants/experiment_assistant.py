__author__ = 'Frederik Diehl'

from apsis.models.experiment import Experiment
from apsis.models.candidate import Candidate
from apsis.utilities.optimizer_utils import check_optimizer

class BasicExperimentAssistant(object):
    """
    This ExperimentAssistant assists with executing experiments.

    It provides methods for getting candidates to evaluate, returning the
    evaluated Candidate and administrates the optimizer.

    Attributes
    ----------

    optimizer: Optimizer
        This is an optimizer implementing the corresponding functions: It
        gets an experiment instance, and returns one or multiple candidates
        which should be evaluated next.

    experiment: Experiment
        The experiment this assistant assists with.
    """

    AVAILABLE_STATUS = ["finished", "pausing", "working"]

    optimizer = None
    experiment = None

    def __init__(self, optimizer, param_defs, minimization=True):
        """
        Initializes the BasicExperimentAssistant.

        Parameters
        ----------

        optimizer: Optimizer instance or string
            This is an optimizer implementing the corresponding functions: It
            gets an experiment instance, and returns one or multiple candidates
            which should be evaluated next.
            Alternatively, it can be a string corresponding to the optimizer,
            as defined by apsis.utilities.optimizer_utils.
        param_defs: dict of ParamDef.
            This is the parameter space defining the experiment.

        minimization=True: bool
            Whether the problem is one of minimization or maximization.
        """
        self.optimizer = optimizer
        self.experiment = Experiment(param_defs, minimization)

    def get_next_candidate(self):
        """
        Returns the Candidate next to evaluate.

        Returns
        -------
        next_candidate: Candidate or None:
            The Candidate object that should be evaluated next. May be None.
        """
        self.optimizer = check_optimizer(self.optimizer)
        if self.experiment.candidates_pending.empty():
            self.experiment.candidates_pending.extend(
                self.optimizer.get_next_candidates(self.experiment))
        return self.experiment.candidates_pending.pop()



    def update(self, candidate, status="finished"):
        """
        Updates the experiment_assistant with the status of an experiment
        evaluation.

        Parameters
        ----------
        candidate: Candidate
            The Candidate object whose status is updated.
        status=finished: string
            A string defining the status change. Can be one of the following:
            - finished: The Candidate is now finished.
            - pausing: The evaluation of Candidate has been paused and can be
                resumed by another worker.
            - working: The Candidate is now being worked on by a worker.
        """
        if status not in self.AVAILABLE_STATUS:
            raise ValueError("status not in %s but %s."
                             %(str(self.AVAILABLE_STATUS), str(status)))

        if not isinstance(candidate, Candidate):
            raise ValueError("candidate %s not a Candidate instance."
                             %str(candidate))

        if status == "finished":
            self.experiment.add_finished(candidate)
            #Also delete all pending candidates from the experiment - we have
            #new data available.
            self.experiment.candidates_pending = []
        elif status == "pausing":
            self.experiment.add_pausing(candidate)
        elif status == "working":
            self.experiment.add_working(candidate)

    def get_best_candidate(self):
        """
        Returns the best candidate to date.

        Returns
        -------
        best_candidate: candidate or None
            Returns a candidate if there is a best one (which corresponds to
            at least one candidate evaluated) or None if none exists.
        """
        return self.experiment.best_candidate