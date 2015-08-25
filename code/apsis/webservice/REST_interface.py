from flask import Flask, request, jsonify
from apsis.assistants.lab_assistant import LabAssistant
from apsis.models.candidate import Candidate, from_dict
from functools import wraps
from apsis.utilities.param_def_utilities import dict_to_param_defs
from apsis.utilities.logging_utils import get_logger
import os

WS_PORT = 5000
CONTEXT_ROOT = ""

app = Flask('apsis')

_logger = None

lAss = None


def start_apsis():
    global lAss, _logger
    _logger = get_logger("REST_interface")
    lAss = LabAssistant()
    app.run(debug=True)
    _logger.info("Finished initialization. Interface running now.")


def exception_handler(func):
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
    if name in lAss.exp_assistants:
        _logger.warning("%s already in names (is %s. Failing the initialization."
                        %(name, lAss.exp_assistants.keys()))
        return "failed"
    optimizer = data_received.get("optimizer", None)
    optimizer_arguments = data_received.get("optimizer_arguments", None)
    minimization = data_received.get("minimization", True)
    param_defs = data_received.get("param_defs", None)
    param_defs = dict_to_param_defs(param_defs)
    lAss.init_experiment(name, optimizer, param_defs, optimizer_arguments,
                         minimization)
    return "success"


@app.route(CONTEXT_ROOT + "/experiments", methods=["GET"])
@exception_handler
def get_all_experiments():
    return lAss.exp_assistants.keys()


@app.route(CONTEXT_ROOT + "/experiments/<experiment_id>", methods=["GET"])
def get_experiment(experiment_id):
   raise NotImplementedError
    #TODO


@app.route(CONTEXT_ROOT + "/experiments/<experiment_id>"
                          "/get_next_candidate", methods=["GET"])
@exception_handler
def get_next_candidate(experiment_id):
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
    data_received = request.get_json()
    status = data_received["status"]
    candidate = from_dict(data_received["candidate"])
    lAss.update(experiment_id, status=status, candidate=candidate)
    return "success"


@app.route(CONTEXT_ROOT + "/experiments/<experiment_id>/candidates",
           methods=["GET"])
@exception_handler
def _get_all_candidates(experiment_id):
    candidates = lAss.get_candidates(experiment_id)
    result = {}
    for r in ["finished", "working", "pending"]:
        result[r] = []
        for i, x in enumerate(candidates[r]):
            result[r].append(x.to_dict())
    return result


def _filter_data(json):
    for k in json:
        if isinstance(json[k], unicode):
            json[k] = str(json[k])
    return json

@app.route(CONTEXT_ROOT + "/experiments/<experiment_id>/fig_results_per_step",
           methods=["GET"])
@exception_handler
def _get_fig_results_per_step(experiment_id):
    return lAss.exp_assistants[experiment_id]._best_result_per_step_dicts(color="b")