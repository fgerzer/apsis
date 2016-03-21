__author__ = 'Frederik Diehl'

import uuid
from apsis.utilities.logging_utils import get_logger
import time

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

    cost : float
        The cost of evaluating the parameter set. This may correspond to
        runtime, cost of ingredients or human attention required.

    worker_information : string
        This is worker-settable information which might be used for
        communicating things necessary for resuming evaluations et cetera. This
        is never touched in apsis.

    last_update_time : float
        The time the last update to this candidate happened.
    """

    cand_id = None
    params = None
    result = None
    cost = None
    worker_information = None
    _logger = None

    last_update_time = None

    def __init__(self, params, cand_id=None, worker_information=None):
        """
        Initializes the unevaluated candidate object.

        Parameters
        ----------
        params : dict of string keys
            A dictionary of parameter value. The keys must correspond to the
            problem definition.
            The dictionary requires one key - and value - per parameter
            defined.
        cand_id : uuid.UUID, optional
            The uuid identifying this candidate. This is used to compare
            candidates over server and client borders.
            Note that this should only be set explicitly if you are
            instantiating an already known candidate with its already known
            UUID. Do not explicitly set the uuid for a new candidate!
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
        self.cand_id = cand_id
        self._logger = get_logger(self, extra_info="cand_id " + str(cand_id))
        self._logger.debug("Initializing new candidate. Params %s, cand_id %s,"
                           "worker_info %s", params, cand_id,
                           worker_information)

        if not isinstance(params, dict):
            self._logger.error("No parameter dict given, received %s instead",
                               params)
            raise ValueError("No parameter dictionary given, received %s "
                             "instead" %params)
        self.params = params
        self.worker_information = worker_information
        self.last_update_time = time.time()
        self._logger.debug("Finished initializing the candidate.")

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
        self._logger.debug("Comparing candidates self (%s) with %s.", self,
                           other)
        if not isinstance(other, Candidate):
            equality = False
        elif self.cand_id == other.cand_id:
            equality = True
        else:
            equality = False
        self._logger.debug("Equality: %s", equality)
        return equality

    def __str__(self):
        """
        Stringifies this Candidate.

        A stringified Candidate is the stringified form of its dict.

        Returns
        -------
        string : string
            The stringified Candidate.

        """
        cand_dict = self.to_dict(do_logging=False)
        string = str(cand_dict)
        return string

    def to_csv_entry(self, delimiter=",", key_order=None):
        """
        Returns a csv entry representing this candidate.

        It is delimited by `delimiter`, and first consists of the id, followed
        by all parameters in the order defined by `key_order`, followed by the
        cost and result.

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
        self._logger.debug("Generating candidate csv entry. Delimiter %s,"
                           "key_order %s", delimiter, key_order)
        if key_order is None:
            key_order = sorted(self.params.keys())
            self._logger.debug("Generated new key order; is %s", key_order)
        string = ""
        string += str(self.cand_id) + delimiter
        for k in key_order:
            string += str(self.params[k]) + delimiter
        string += str(self.cost) + delimiter
        string += str(self.result)
        self._logger.debug("csv entry is %s", string)
        return string

    def to_dict(self, do_logging=True):
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
            "cost" : float or None
                The cost of evaluating the Candidate
            "worker_information" : any jsonable or None
                Client-settable worker information.
        """
        if do_logging:
            self._logger.debug("Converting cand to dict.")
        d = {"cand_id": self.cand_id,
             "params": self._param_defs_to_dict(do_logging=do_logging),
             "result": self.result,
             "cost": self.cost,
             "last_update_time": self.last_update_time,
             "worker_information": self.worker_information}
        if do_logging:
            self._logger.debug("Generated dict %s", d)
        return d

    def _param_defs_to_dict(self, do_logging=True):
        """
        Returns a parameter definition dictionary representation.

        Returns
        -------
        d : dict
            Dictionary of the parameters.
        """
        if do_logging:
            self._logger.debug("Converting param_def to dict.")
        d = {}
        for k in self.params.keys():
            d[k] = self.params[k]
        if do_logging:
            self._logger.debug("param_def dict is %s", d)
        return d

global_logger = get_logger("models.Candidate")


def from_dict(d):
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
    global_logger.debug("Constructing new candidate from dict %s.", d)
    cand_id = None
    if "cand_id" in d:
        cand_id = d["cand_id"]
    c = Candidate(d["params"], cand_id=cand_id)
    c.result = d.get("result", None)
    c.cost = d.get("cost", None)
    c.last_update_time = d.get("last_update_time")
    c.worker_information = d.get("worker_information", None)
    global_logger.debug("Constructed candidate is %s", c)
    return c
