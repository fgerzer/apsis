from flask import Flask, request, jsonify
from flask_negotiate import consumes, produces
from apsis.assistants.lab_assistant import LabAssistant
#from apsis.models.parameter_definition import *
from apsis.models.candidate import Candidate, from_dict
#from apsis.models.experiment import Experiment
import multiprocessing
from multiprocessing import reduction
import traceback
from apsis.utilities.param_def_utilities import dict_to_param_defs
#import json

import logging

WS_PORT = 5000
CONTEXT_ROOT = ""

logging.basicConfig(level=logging.DEBUG)

app = Flask('apsis')

lAss = None

def start_apsis():
    global lAss
    lAss = LabAssistant()
    app.run(debug=True)
    print("Started app. Initialized LAss.")

@app.route(CONTEXT_ROOT + "/", methods=["GET"])
#@produces('application/json')
def overview_page():
    """
    This will, later, become an overview over the experiment.
    """
    experiments = lAss.exp_assistants.keys()
    str(experiments)


@app.route(CONTEXT_ROOT + "/experiments", methods=["POST"])
#@consumes('application/json')
#@produces('application/json')
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
    try:
        data_received = request.get_json()
        data_received = _filter_data(data_received)
        name = data_received.get("name", None)
        if name in lAss.exp_assistants:
            return "Error: %s already exists." %name
        optimizer = data_received.get("optimizer", None)
        optimizer_arguments = data_received.get("optimizer_arguments", None)
        minimization = data_received.get("minimization", True)
        param_defs = data_received.get("param_defs", None)
        param_defs = dict_to_param_defs(param_defs)
        lAss.init_experiment(name, optimizer, param_defs, optimizer_arguments,
                             minimization)
        return "Experiment initialized successfully."
    except:
        return str(traceback.print_exc() + "\nInitialization failed.")

@app.route(CONTEXT_ROOT + "/experiments", methods=["GET"])
def get_all_experiments():
    print("Got get all exps")
    return jsonify(result=lAss.exp_assistants.keys())

@app.route(CONTEXT_ROOT + "/experiments/<experiment_id>", methods=["GET"])
def get_experiment(experiment_id):
   pass
    #TODO

@app.route(CONTEXT_ROOT + "/experiments/<experiment_id>"
                          "/get_next_candidate", methods=["GET"])
def get_next_candidate(experiment_id):
    result_cand = lAss.exp_assistants[experiment_id].get_next_candidate()
    if result_cand is None:
        result = "None"
    else:
        result = result_cand.to_dict()
    return jsonify(result=result)

@app.route(CONTEXT_ROOT + "/experiments/<experiment_id>"
                          "/get_best_candidate", methods=["GET"])
def get_best_candidate(experiment_id):
    result_cand = lAss.exp_assistants[experiment_id].get_best_candidate()
    if result_cand is None:
        result = "None"
    else:
        result = result_cand.to_dict()
    return jsonify(result=result)

@app.route(CONTEXT_ROOT + "/experiments/<experiment_id>"
                          "/update", methods=["POST"])
def update(experiment_id):
    data_received = request.get_json()
    status = data_received["status"]
    candidate = from_dict(data_received["candidate"])
    #lAss.update(status=status, candidate=candidate)
    lAss.exp_assistants[experiment_id].update(status=status, candidate=candidate)
    return "Success"

@app.route(CONTEXT_ROOT + "/experiments/<experiment_id>/candidates",
           methods=["GET"])
def _get_all_candidates(experiment_id):
    candidates = lAss.get_candidates(experiment_id)
    result = {}
    for r in ["finished", "working", "pending"]:
        result[r] = []
        for i, x in enumerate(candidates[r]):
            result[r].append(x.to_dict())
    return jsonify(result=result)


def _send_msg_lab(msg):
    #msg["result_queue"] = reduction.reduce_connection(msg["result_queue"])
    lqueue.put(msg)

def _filter_data(json):
    for k in json:
        if isinstance(json[k], unicode):
            json[k] = str(json[k])
    return json
#
# @app.route(CONTEXT_ROOT + "/experiments/<experiment_id>/working", methods=["POST"])
# @consumes('application/json')
# @produces('application/json')
# def working(experiment_id):
#     """
#     POST method to post new information from a worker targetting at the cores
#     working method. Needs to be given the experiment_id of the experiment
#     to which the result belongs.
#
#     POST DATA
#     ---------
#     candidate: dict
#         dict representing Candidate object.
#     status: String
#         indicating status of this candidate: "finished", "working",...
#     [worker_id]: String
#         A string id describing a worker.
#     [can_be_killed=False]: boolean
#         If this worker can be killed.
#     """
#     logging.debug("POST experiments/<id>/working for experiment id " + experiment_id)
#
#     request_body = request.get_json()
#
#     if(request_body is None):
#         return "ERORR (1) - request body empty", 500
#
#     #mandatory
#     serialized_candidate = request_body.get('candidate', None)
#     candidate_status = request_body.get('status', None)
#
#     #optional
#     worker_id = request_body.get('worker_id', None)
#     can_be_killed = request_body.get('can_be_killed', False)
#
#     if(serialized_candidate is not None and candidate_status is not None):
#         deserialized_candidate = Candidate.from_dict(serialized_candidate)
#         logging.debug("Deserialized candidate in state " +
#                       str(type(candidate_status)) +  ": " +
#                       str(candidate_status) + " "
#                       + str(deserialized_candidate))
#
#         #TODO use experiment id to register working in the appropriate experiment
#         #TODO call to OptimizationCoreInterface.working
#         #TODO return the continue value for the experiment posted
#         return jsonify({"continue": True}), 200
#
#     else:
#         return "ERORR (2) - candidate object was not given", 500
#
#
# @app.route(CONTEXT_ROOT + "/experiments/<experiment_id>/candidate", methods=["GET"])
# @produces('application/json')
# def next_candidate(experiment_id):
#     logging.debug("GET experiments/<id>/candidate for experiment id " +
#                   experiment_id)
#
#     #TODO use experiment id to obtain next candidate from
#     #sample code for testing follows
#     test_point = [1, 1, 1]
#     for i in range(0, len(test_point)):
#         test_point[i] = random.gauss(0, 10)
#     test_candidate = Candidate(test_point)
#
#     #return test candidate as dict and then bring to json
#     return jsonify(test_candidate.__dict__), 200
#
#
# logging.info("APSIS web service starting at " + str(WS_PORT))
# app.run(debug=True, port=WS_PORT)
#
#
