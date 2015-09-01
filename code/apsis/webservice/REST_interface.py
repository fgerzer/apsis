from flask import Flask, request, jsonify
from apsis.assistants.lab_assistant import LabAssistant, ValidationLabAssistant
from apsis.models.candidate import Candidate, from_dict
from functools import wraps
from apsis.utilities.param_def_utilities import dict_to_param_defs
from apsis.utilities.logging_utils import get_logger
import os
import signal

WS_PORT = 5000
CONTEXT_ROOT = ""

app = Flask('apsis')

_logger = None

lAss = None

def set_exit(_signo, _stack_frame):
    """
    Sets the exit for the lab assistant.
    """
    _logger.warning("Shutting down apsis server, due to signal %s with "
                    "stackframe %s" %(_signo, _stack_frame))
    lAss.set_exit()
    exit()

signal.signal(signal.SIGINT, set_exit)


def start_apsis(validation=False, cv=5):
    """
    Starts apsis.

    Initializes logger, LabAssistant and the REST app.
    """
    global lAss, _logger
    _logger = get_logger("REST_interface")
    if validation:
        lAss = ValidationLabAssistant(cv=cv)
    else:
        lAss = LabAssistant()
    app.run(debug=True)
    _logger.info("Finished initialization. Interface running now.")


def exception_handler(func):
    """
    This wrapper is used to handle jsonifying and exceptions.

    Specficially, it tries to jsonify the function, with the result being
    written to the "result" field. Any failure is catched and logged.
    If failed, "result" is set to "failed".
    """
    @wraps(func)
    def handle_exception(*args, **kwargs):
        try:
            return jsonify(result=func(*args, **kwargs))
        except:
            _logger.exception("Exception while handling the answer. Catching "
                              "to prevent server crash.")
            return jsonify(result="failed")
    return handle_exception


@app.route(CONTEXT_ROOT + "/", methods=["GET"])
@exception_handler
def overview_page():
    """
    This will, later, become an overview over the experiment.
    """
    experiments = lAss.exp_assistants.keys()
    str(experiments)
    raise NotImplementedError
    #TODO


@app.route(CONTEXT_ROOT + "/experiments", methods=["POST"])
@exception_handler
def init_experiment():
    """
    This initializes a single experiment.

    The json-data to be sent should be a dictionary of the following format:
    {
    "name": string
        The name of the experiment. Must be unique.
    "optimizer": string
        The optimizer to be used or None to automatically choose one.
    "param_defs": list of ParamDef dicts.
        Each entry of this must be in the following format:
        {
        "type": string
            The type of the parameter definition
        "<parameter_name>": <parameter_type>
            Where each of the necessary parameters for the ParamDef must be
            included.
        }
    "optimizer_arguments": dict
        Dictionary of the arguments of the optimizer, in key-value pairs.
    "minimization": bool, optional
        Whether the problem is one of minimization or maximization. Default
        is minimization.
    }
    """
    data_received = request.get_json()
    data_received = _filter_data(data_received)
    name = data_received.get("name", None)
    exp_id = data_received.get("exp_id", None)
    notes = data_received.get("notes", None)
    if exp_id in lAss.exp_assistants:
        _logger.warning("%s already in exp_ids. (is %s). Failing the initialization."
                        %(exp_id, lAss.exp_assistants.keys()))
        return "failed"
    optimizer = data_received.get("optimizer", None)
    optimizer_arguments = data_received.get("optimizer_arguments", None)
    minimization = data_received.get("minimization", True)
    param_defs = data_received.get("param_defs", None)
    param_defs = dict_to_param_defs(param_defs)
    exp_id = lAss.init_experiment(name, optimizer, param_defs,
                              exp_id, notes, optimizer_arguments, minimization)
    print("EXP_ID: " + str(exp_id))
    print(type(exp_id))
    return exp_id

@app.route(CONTEXT_ROOT + "/experiments", methods=["GET"])
@exception_handler
def get_all_experiments():
    """
    This returns all experiment IDs.
    """
    return lAss.exp_assistants.keys()


@app.route(CONTEXT_ROOT + "/experiments/<experiment_id>", methods=["GET"])
@exception_handler
def get_experiment(experiment_id):
    """
    This will, later, return more details for a single experiment.
    """
    raise NotImplementedError
    #TODO return whole experiment.


@app.route(CONTEXT_ROOT + "/experiments/<experiment_id>"
                          "/get_next_candidate", methods=["GET"])
@exception_handler
def get_next_candidate(experiment_id):
    """
    Returns the next candidate for a specific experiment.

    Parameters
    ----------
    experiment_id : string
        The exp_id of the experiment for which the candidate should be
        returned.

    Returns
    -------
    result : Candidate, None or "failed".
        Returns either a Candidate (if successful), None if none is available
        or possibly "failed" if the request failed.
    """
    result_cand = lAss.get_next_candidate(experiment_id)
    if result_cand is None:
        result = "failed"
    else:
        result = result_cand.to_dict()
    return result


@app.route(CONTEXT_ROOT + "/experiments/<experiment_id>"
                          "/get_best_candidate", methods=["GET"])
@exception_handler
def get_best_candidate(experiment_id):
    """
    Returns the best finished candidate for an experiment.

    Parameters
    ----------
    exp_id : string
        The id of the experiment to return.

    Returns
    -------
    best_candidate : dict as a candidate representation
        Dictionary candidate representation (see get_next_candidate for the
        exact format). May be None or "failed" if no such candidate exists.
    """
    result_cand = lAss.get_best_candidate(experiment_id)
    if result_cand is None:
        result = "failed"
    else:
        result = result_cand.to_dict()
    return result


@app.route(CONTEXT_ROOT + "/experiments/<experiment_id>"
                          "/update", methods=["POST"])
@exception_handler
def update(experiment_id):
    """
    Updates the result of the candidate.

    Parameters
    ----------
    exp_id : string
        The id of the experiment to return.
    json : json dict
        Contains two elements.
        "candidate" : dict representing a candidate
            Represents a candidate. Usually a modified candidate received from
             get_next_candidate.
            It consists of the following fields:
            "cost" : float or None
                The cumulative cost of the evaluations of this candidate.
                Must be set by the worker. Default is None, representing no
                cost being set.
            "params" : dict of parameters
                The parameter values this candidate has. The format is
                analogous to the parameter defintion of init_experiment,
                with each entry being an acceptable value according to
                param_def.
            "id" : string
                An id uniquely identifying this candidate.
            "worker_information" : arbitrary
                A field usable for setting worker information, for example a
                directory in which intermediary results are stored. Any
                json-able information can be stored in it (though, since
                it's transferred via network, it is probably better to keep it
                fairly small), and apsis guarantees never to change it.
                By default, it's None.
            "result" : float
                The result of the process we want to optimize.
                Is None by default
        "status" : string
            One of "finished", "working" and "pausing".
            "finished": The evaluation is finished.
            "working": The evaluation is still in progress. Later, it will be
            used to ensure that the worker is still working, allowing us to
            reschedule the candidate to other workers if necessary.
            "pausing": Signals that this candidate has paused the execution,
            meaning that we are allowed to reschedule it to another worker.

    Returns
    -------
    result : string
        Returns "success" iff successful, "failed" otherwise.
    """
    data_received = request.get_json()
    status = data_received["status"]
    candidate = from_dict(data_received["candidate"])
    lAss.update(experiment_id, status=status, candidate=candidate)
    return "success"


@app.route(CONTEXT_ROOT + "/experiments/<experiment_id>/candidates",
           methods=["GET"])
@exception_handler
def _get_all_candidates(experiment_id):
    """
    Returns the candidates for an experiment.

    Parameters
    ----------
    exp_id : string
        The id of the experiment to return.

    Returns
    -------
    candidates : dict of lists
        Returns a dictionary of three lists of candidates.
        Each of the lists contains dictionary candidate representation
        (see get_next_candidate for the exact format). Each list may be
        empty.
        The three lists are:
        "finished": The list of finished candidates.
        "workign": The list of candidates on which workers are currently
        working.
        "pending": The list of not-yet finished candidates on which no
        worker is currently working.
        May return None or "failed" if failed.
    """
    candidates = lAss.get_candidates(experiment_id)
    result = {}
    for r in ["finished", "working", "pending"]:
        result[r] = []
        for i, x in enumerate(candidates[r]):
            result[r].append(x.to_dict())
    return result


def _filter_data(json):
    """
    Filters json data.

    More specifically, it converts strings to unicode in all json fields.
    """
    for k in json:
        if isinstance(json[k], unicode):
            json[k] = str(json[k])
    return json

@app.route(CONTEXT_ROOT + "/experiments/<experiment_id>/fig_results_per_step",
           methods=["GET"])
@exception_handler
def _get_fig_results_per_step(experiment_id):
    """
    Currently unused.
    """
    return lAss.exp_assistants[experiment_id]._best_result_per_step_dicts(color="b")