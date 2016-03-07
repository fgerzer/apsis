from flask import Flask, request, jsonify, render_template
from apsis.assistants.lab_assistant import LabAssistant
from apsis.models.candidate import Candidate, from_dict
from functools import wraps
from apsis.utilities.param_def_utilities import dict_to_param_defs
import os
import sys
import signal
import urllib
import StringIO
import datetime
import time
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from apsis.utilities import file_utils
from apsis.utilities import logging_utils
import traceback

CONTEXT_ROOT = ""

app = Flask('apsis')

_logger = None

lAss = None

should_fail_deadly = False

def set_exit(_signo, _stack_frame):
    """
    Sets the exit for the lab assistant.
    """
    _logger.warning("Shutting down apsis server, due to signal %s with "
                    "stackframe %s" %(_signo, _stack_frame))
    lAss.set_exit()
    sys.exit()

signal.signal(signal.SIGINT, set_exit)


def start_apsis(port=5000, continue_path=None, fail_deadly=False):
    """
    Starts apsis.

    Initializes logger, LabAssistant and the REST app.
    """
    global lAss, _logger
    _logger = logging_utils.get_logger("webservice.REST_interface")
    if fail_deadly:
        print("WARNING! Fail deadly is active. Make sure you know what you do."
              "State of the program might be lost at any time, and the "
              "program might crash unexpectedly.")
        _logger.warning("WARNING! Fail deadly is active. Make sure you know "
                        "what you do. State of the program might be lost at "
                        "any time, and the program might crash unexpectedly.")



    if continue_path:
        write_dir = continue_path
    else:
        if os.name == "nt":
            write_dir = os.path.relpath("APSIS_WRITING")
        else:
            write_dir = "/tmp/APSIS_WRITING"
        date_name = datetime.datetime.utcfromtimestamp(
                time.time()).strftime("%Y-%m-%d_%H.%M.%S")
        write_dir = os.path.join(write_dir, date_name)
    global should_fail_deadly
    should_fail_deadly = fail_deadly

    file_utils.ensure_directory_exists(write_dir)
    lAss = LabAssistant(write_dir=write_dir)
    app.run(host='0.0.0.0', debug=False, port=port)
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
        except Exception as e:
            _logger.exception("Exception while handling the answer. Exception "
                              "is %s", e)

            if should_fail_deadly:
                print(e)
                request.environ.get('werkzeug.server.shutdown')()
                lAss.set_exit()
                raise RuntimeError("Exception raised and fail_deadly active."
                                " Raising general exception. Original "
                                "exception is " + str(e))

            return jsonify(result="failed")
    return handle_exception


@app.route(CONTEXT_ROOT + "/", methods=["GET"])
@exception_handler
def overview_page():
    """
    This will, later, become an overview over the experiment.
    """
    _logger.log(5, "Returning overview page.")
    return render_template("overview.html", experiments=lAss.get_ids())


@app.route(CONTEXT_ROOT + "/c/experiments", methods=["POST"])
@exception_handler
def client_init_experiment():
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
    _logger.debug("Initializing experiment. Request is %s, json %s", request,
                  request.json)
    data_received = request.get_json()
    data_received = _filter_data(data_received)
    name = data_received.get("name", None)
    exp_id = data_received.get("exp_id", None)
    notes = data_received.get("notes", None)
    if lAss.contains_id(exp_id):
        _logger.warning("%s already in exp_ids. (Exp_ids known are %s). "
                        "Failing the initialization."
                        %(exp_id, lAss.get_ids()))
        return "failed"
    optimizer = data_received.get("optimizer", None)
    optimizer_arguments = data_received.get("optimizer_arguments", None)
    minimization = data_received.get("minimization", True)
    param_defs = data_received.get("param_defs", None)
    param_defs = dict_to_param_defs(param_defs)
    _logger.debug("Initializing experiment.")
    exp_id = lAss.init_experiment(name, optimizer, param_defs,
                              exp_id, notes, optimizer_arguments, minimization)
    _logger.info("Initialized new experiment of name %s. exp_id is %s",
                 name, exp_id)
    return exp_id


@app.route(CONTEXT_ROOT + "/c/experiments", methods=["GET"])
@exception_handler
def client_get_all_experiments():
    """
    This returns all experiment IDs.
    """
    exp_ids = lAss.get_ids()
    _logger.debug("Asked for all experiment ids. Returning %s", exp_ids)
    return exp_ids


@app.route(CONTEXT_ROOT + "/c/experiments/<experiment_id>", methods=["GET"])
@exception_handler
def client_get_experiment(experiment_id):
    """
    This will, later, return more details for a single experiment.
    """
    _logger.debug("Asked for experiment with id %s", experiment_id)
    experiment_dict = lAss.get_experiment_as_dict(experiment_id)
    _logger.debug("Returned exp_dict %s", experiment_dict)
    return experiment_dict


@app.route(CONTEXT_ROOT + "/experiments/<experiment_id>", methods=["GET"])
def get_experiment(experiment_id):
    """
    This will, later, return more details for a single experiment.
    """
    _logger.debug("Asked for experiment id %s. This returns the page.",
                  experiment_id)
    exp_dict = lAss.get_experiment_as_dict(experiment_id)
    param_defs = exp_dict["parameter_definitions"]
    finished_candidates_string = exp_dict["candidates_finished"]
    pending_candidates_string = exp_dict["candidates_pending"]
    working_candidates_string = exp_dict["candidates_working"]
    best_candidate_string = exp_dict["best_candidate"]
    fig = lAss.get_plot_result_per_step(experiment_id)
    fig.autofmt_xdate()
    canvas=FigureCanvas(fig)
    png_output = StringIO.StringIO()
    canvas.print_png(png_output)
    png_output = png_output.getvalue().encode("base64")

    _logger.debug("Rendering template")
    templ = render_template("experiment.html",
                           exp_name=exp_dict["name"],
                           exp_id=exp_dict["exp_id"],
                           minimization=exp_dict["minimization_problem"],
                           param_defs=param_defs,
                           finished_candidates_string=finished_candidates_string,
                           pending_candidates_string=pending_candidates_string,
                           working_candidates_string=working_candidates_string,
                           result_per_step=urllib.quote(png_output.rstrip('\n')),
                           best_candidate_string=best_candidate_string
                           )
    _logger.log(5, "Returning template %s", templ)
    return templ


@app.route(CONTEXT_ROOT + "/c/experiments/<experiment_id>"
                          "/get_next_candidate", methods=["GET"])
@exception_handler
def client_get_next_candidate(experiment_id):
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
    _logger.debug("Should return next candidate for %s", experiment_id)
    result_cand = lAss.get_next_candidate(experiment_id)
    if result_cand is None:
        _logger.debug("No next candidate available. Failing.")
        result = "failed"
    else:
        result = result_cand.to_dict()
    _logger.debug("Returning next cand %s", result)
    return result


@app.route(CONTEXT_ROOT + "/c/experiments/<experiment_id>"
                          "/get_best_candidate", methods=["GET"])
@exception_handler
def client_get_best_candidate(experiment_id):
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
    _logger.debug("Returning best candidate for %s", experiment_id)
    result_cand = lAss.get_best_candidate(experiment_id)
    if result_cand is None:
        _logger.debug("No best candidate available. Returning failed.")
        result = "failed"
    else:
        result = result_cand.to_dict()
    _logger.debug("Returning best candidate %s", result)
    return result


@app.route(CONTEXT_ROOT + "/c/experiments/<experiment_id>"
                          "/update", methods=["POST"])
@exception_handler
def client_update(experiment_id):
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
    _logger.debug("Updating client. request is %s, json %s", request,
                  request.json)
    data_received = request.get_json()
    status = data_received["status"]
    candidate = from_dict(data_received["candidate"])
    lAss.update(experiment_id, status=status, candidate=candidate)
    _logger.debug("Updated lAss.")
    return "success"


@app.route(CONTEXT_ROOT + "/c/experiments/<experiment_id>/candidates",
           methods=["GET"])
@exception_handler
def client_get_all_candidates(experiment_id):
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
    _logger.debug("Ready to return all candidates for %s", experiment_id)
    candidates = lAss.get_candidates(experiment_id)
    result = {}
    for r in ["finished", "working", "pending"]:
        result[r] = []
        for i, x in enumerate(candidates[r]):
            result[r].append(x.to_dict())
    _logger.debug("Returning all candidates %s", result)
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