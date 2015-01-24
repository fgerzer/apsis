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

    params : dict of string keys
        A dictionary of parameter value. The keys must correspond to the
        problem definition.
        The dictionary requires one key - and value - per parameter defined.

    result : float
        The results of evaluating the parameter set. This value is optimized
        over.

    cost : float
        The cost of evaluating the parameter set. This may correspond to
        runtime, cost of ingredients or human attention required.

    worker_information : string
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
        params : dict of string keys
            A dictionary of parameter value. The keys must correspond to the
            problem definition.
            The dictionary requires one key - and value - per parameter defined.
        worker_information : string, optional
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
        Compares two Candidate instances.

        Two Candidate instances are defined as being equal iff their params
        vectors are equal. A non-Candidate instance is never equal to a
        Candidate.

        Parameters
        ----------
        other :
            The object to compare this Candidate instance to.

        Returns
        -------
        equality : bool
            True iff other is a Candidate instance and their params are
            identical.
        """

        if not isinstance(other, Candidate):
            return False

        if self.params == other.params:
            return True
        return False

    def __str__(self):
        """
        Stringifies this Candidate.

        A stringified Candidate is of the form:
        Candidate
        params: XXX
        cost: XXX
        result XXX

        Returns
        -------
        string : string
            The stringified Candidate.

        """
        string = "Candidate\n"
        string += "params: " + str(self.params) + "\n"
        if self.cost is not None:
            string += "cost: " + str(self.cost) + "\n"
        string += "result: " + str(self.result) + "\n"
        return string

    def to_csv_entry(self, delimiter=",", key_order=None):
        """
        Returns a csv entry representing this candidate.

        It is delimited by `delimiter`, and first consists of all parameters
        in the order defined by `key_order`, followed by the cost and result.

        Parameters
        ----------
            delimiter : string, optional
                The string delimiting csv entries
            key_order : list of param names, optional
                A list defining the order of keys written to csv. If None, the
                order will be set by sorting the keys.

        Returns
        -------
            string : string
                The (one-line) string representing this Candidate as a csv line
        """
        if key_order is None:
            key_order = sorted(self.params.keys())
        string = ""
        for k in key_order:
            string += str(self.params[k]) + delimiter
        string += str(self.cost) + delimiter
        string += str(self.result)
        return string

    def to_dict(self):
        """
        EXPERIMENTAL
        """
        d = {}
        d["params"] = self._param_defs_to_dict()
        d["result"] = self.result
        d["cost"] = self.cost
        d["worker_information"] = self.worker_information

        return d

    def _param_defs_to_dict(self):
        """
        EXPERIMENTAL
        """
        d = {}
        for k in self.params.keys():
            d[k] = self.params[k]
        return d

def from_dict(dict):
    """
    EXPERIMENTAL
    """
    c = Candidate(dict["params"])
    c.result = dict.get("result", None)
    c.cost = dict.get("cost", None)
    c.worker_information = dict.get("worker_information", None)
    return c