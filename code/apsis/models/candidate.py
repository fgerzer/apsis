__author__ = 'Frederik Diehl'

import uuid

class Candidate(object):
    """
    A Candidate is a dictionary of parameter values, which should - or have
    been - evaluated.

    A Candidate object can be seen as a single iteration of the experiment.
    It is first generated as a suggestion of which parameter set to evaluate
    next, then updated with the result and cost of the evaluation.

    Attributes
    ----------

    id : uuid.UUID
        The uuid identifying this candidate. This is used to compare candidates
        over server and client borders.

    params : dict of string keys
        A dictionary of parameter value. The keys must correspond to the
        problem definition.
        The dictionary requires one key - and value - per parameter defined.

    result : float
        The results of evaluating the parameter set. This value is optimized
        over.

    failed : bool
        Whether the evaluation has been successful. Default is false.

    cost : float
        The cost of evaluating the parameter set. This may correspond to
        runtime, cost of ingredients or human attention required.

    worker_information : string
        This is worker-settable information which might be used for
        communicating things necessary for resuming evaluations et cetera. This
        is never touched in apsis.
    """

    cand_id = None
    params = None
    result = None
    cost = None
    failed = None
    worker_information = None

    def __init__(self, params, cand_id=None, worker_information=None):
        """
        Initializes the unevaluated candidate object.

        Parameters
        ----------
        params : dict of string keys
            A dictionary of parameter value. The keys must correspond to the
            problem definition.
            The dictionary requires one key - and value - per parameter defined.
        cand_id : uuid.UUID, optional
            The uuid identifying this candidate. This is used to compare candidates
            over server and client borders.
            Note that this should only be set explicitly if you are instantiating
             an already known candidate with its already known UUID. Do not
             explicitely set the uuid for a new candidate!
        worker_information : string, optional
            This is worker-settable information which might be used for
            communicating things necessary for resuming evaluations et cetera.

        Raises
        ------
        ValueError
            Iff params is not a dictionary.
        """
        if cand_id is None:
            cand_id = uuid.uuid4().hex
        self.id = cand_id
        if not isinstance(params, dict):
            raise ValueError("No parameter dictionary given.")
        self.failed = False
        self.params = params
        self.worker_information = worker_information

    def __eq__(self, other):
        """
        Compares two Candidate instances.

        Two Candidate instances are defined as being equal iff their ids
        are equal. A non-Candidate instance is never equal to a
        Candidate.

        Parameters
        ----------
        other :
            The object to compare this Candidate instance to.

        Returns
        -------
        equality : bool
            True iff other is a Candidate instance and their ids are equal.
        """

        if not isinstance(other, Candidate):
            return False

        if self.cand_id == other.cand_id:
            return True
        return False

    def __str__(self):
        """
        Stringifies this Candidate.

        A stringified Candidate is of the form:
        Candidate
        id: XXX
        params: XXX
        cost: XXX
        result XXX

        Returns
        -------
        string : string
            The stringified Candidate.

        """
        string = "Candidate\n"
        string += "id: %s\n" %self.id
        string += "params: %s\n" %str(self.params)
        if self.cost is not None:
            string += "cost: %s\n" %self.cost
        string += "result: %s\n" %str(self.result)
        string += "failed: %s\n" %str(self.failed)
        return string

    def to_csv_entry(self, delimiter=",", key_order=None):
        """
        Returns a csv entry representing this candidate.

        It is delimited by `delimiter`, and first consists of the id, followed
        by all parameters in the order defined by `key_order`, followed by the
        cost, the results and the failure state.

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
        string += str(self.id) + delimiter
        for k in key_order:
            string += str(self.params[k]) + delimiter
        string += str(self.cost) + delimiter
        string += str(self.result) + delimiter
        string += str(self.failed)
        return string

    def to_dict(self):
        """
        Converts this candidate to a dictionary.

        Returns
        -------
        d : dictionary
            Contains the following key/value pairs:
            "id" : string
                The id of the candidate.
            "params" : dict
                This dictionary contains one entry for each parameter,
                each with the string name as key and the value as value.
            "result" : float or None
                The result of the Candidate
            "failed" : bool
                Whether the evaluation failed.
            "cost" : float or None
                The cost of evaluating the Candidate
            "worker_information" : any jsonable or None
                Client-settable worker information.
        """
        d = {"id": self.id,
             "params": self._param_defs_to_dict(),
             "result": self.result,
             "failed": self.failed,
             "cost": self.cost,
             "worker_information": self.worker_information}
        return d

    def _param_defs_to_dict(self):
        """
        Returns a parameter definition dictionary representation.

        Returns
        -------
        d : dict
            Dictionary of the parameters.
        """
        d = {}
        for k in self.params.keys():
            d[k] = self.params[k]
        return d


def from_dict(cand_dict):
    """
    Builds a new candidate from a dictionary.

    Parameters
    ----------
    cand_dict : dictionary
        Uses the same format as in Candidate.to_dict.

    Returns
    -------
    c : Candidate
        The corresponding candidate.
    """
    cand_id = None
    if "id" in cand_dict:
        cand_id = cand_dict["id"]
    c = Candidate(cand_dict["params"], cand_id=cand_id)
    c.result = cand_dict.get("result", None)
    c.failed = cand_dict.get("failed", False)
    c.cost = cand_dict.get("cost", None)
    c.worker_information = cand_dict.get("worker_information", None)
    return c
