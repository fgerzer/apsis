__author__ = 'Frederik Diehl'


class Candidate(object):
    """
    A Candidate is a dictionary of parameter values, which should - or have
    been - evaluated.

    A Candidate object can be seen as a single iteration of the experiment.
    It is first generated as a suggestion of which parameter set to evaluate
    next, then updated with the result and cost of the evaluation.

    Attributes
    ----------

    params: dict of string keys
        A dictionary of parameter value. The keys must correspond to the
        problem definition.
        The dictionary requires one key - and value - per parameter defined.

    result: float
        The results of evaluating the parameter set. This value is optimized
        over.

    cost: float
        The cost of evaluating the parameter set. This may correspond to
        runtime, cost of ingredients or human attention required.

    worker_information: string
        This is worker-settable information which might be used for
        communicating things necessary for resuming evaluations et cetera. This
        is never touched in apsis.
    """

    params = None
    result = None
    cost = None
    worker_information = None

    def __init__(self, params, worker_information=None):
        """
        Initializes the unevaluated candidate object.

        Parameters
        ----------
        params: dict of string keys
            A dictionary of parameter value. The keys must correspond to the
            problem definition.
            The dictionary requires one key - and value - per parameter defined.
        worker_information: string
            This is worker-settable information which might be used for
            communicating things necessary for resuming evaluations et cetera.

        Raises
        ------
        ValueError
            Iff params is not a dictionary.
        """
        if not isinstance(params, dict):
            raise ValueError("No parameter dictionary given.")
        self.params = params
        self.worker_information = worker_information

    def __eq__(self, other):
        """
        Compares two Canidate instances.

        Two Candidate instances are defined as being equal iff their params
        vectors are equal. A non-Candidate instance is never equal to a
        Candidate.

        Parameters
        ----------
        other: object
            The object to compare this Candidate instance to.

        Returns
        -------
        equality: bool
            True iff other is a Candidate instance and their params are
            identical.
        """

        if not isinstance(other, Candidate):
            return False

        if self.params == other.params:
            return True
        return False

    def __str__(self):
        string = "Candidate\n"
        string += "params: " + str(self.params) + "\n"
        if self.cost is not None:
            string += "cost: " + str(self.cost) + "\n"
        string += "result: " + str(self.result) + "\n"
        return string