__author__ = 'Frederik Diehl'

from apsis.models.experiment import Experiment
from apsis.models.candidate import Candidate
from apsis.utilities.optimizer_utils import check_optimizer
import matplotlib.pyplot as plt

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
    optimizer_arguments: dict
        These are arguments for the optimizer. Refer to their documentation
        as to which are available.

    experiment: Experiment
        The experiment this assistant assists with.
    """

    AVAILABLE_STATUS = ["finished", "pausing", "working"]

    optimizer = None
    optimizer_arguments = None
    experiment = None

    def __init__(self, name, optimizer, param_defs, optimizer_arguments=None,
                 minimization=True):
        """
        Initializes the BasicExperimentAssistant.

        Parameters
        ----------
        name: string
            The name of the experiment. This does not have to be unique, but is
            for human orientation.
        optimizer: Optimizer instance or string
            This is an optimizer implementing the corresponding functions: It
            gets an experiment instance, and returns one or multiple candidates
            which should be evaluated next.
            Alternatively, it can be a string corresponding to the optimizer,
            as defined by apsis.utilities.optimizer_utils.
        param_defs: dict of ParamDef.
            This is the parameter space defining the experiment.
        optimizer_arguments=None: dict
            These are arguments for the optimizer. Refer to their documentation
            as to which are available.
        minimization=True: bool
            Whether the problem is one of minimization or maximization.
        """
        self.optimizer = optimizer
        self.optimizer_arguments = optimizer_arguments
        self.experiment = Experiment(name, param_defs, minimization)

    def get_next_candidate(self):
        """
        Returns the Candidate next to evaluate.

        Returns
        -------
        next_candidate: Candidate or None:
            The Candidate object that should be evaluated next. May be None.
        """
        self.optimizer = check_optimizer(self.optimizer)
        if not self.experiment.candidates_pending:
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

class PrettyExperimentAssistant(BasicExperimentAssistant):
    """
    A 'prettier' version of the experiment assistant, mostly through plots.
    """


    def plot_result_per_step(self):

        this_plot = plt.figure()
        x, step_eval, step_best = self._best_result_per_step_data()
        plt.plot(x, step_best, label="%s, best result"
                                     %str(self.experiment.name))
        plt.scatter(x, step_eval, label="%s, current result"
                                        %str(self.experiment.name))
        plt.xlabel("steps")
        plt.ylabel("result")
        if self.experiment.minimization_problem:
            plt.legend(loc='upper right')
        else:
            plt.legend(loc='upper left')
        plt.show()

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