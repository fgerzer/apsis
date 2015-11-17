__author__ = 'Frederik Diehl'
import apsis
from apsis.assistants.lab_assistant import LabAssistant
import apsis.webservice.REST_interface as REST_interface
import time
from apsis.models.parameter_definition import *
import multiprocessing
from multiprocessing import reduction
from apsis_client.apsis_connection import Connection

from apsis.utilities.param_def_utilities import param_defs_to_dict

import apsis.models.parameter_definition as pd
import time

#from apsis.webservice.REST_interface import app

def scaled_branin_hoo(x, y, z="C"):
    a = 1
    b = 5.1/(4*math.pi**2)
    c = 5.0/math.pi
    r = 6
    s = 10
    t = 1/(8*math.pi)
    f1 = a*(y - b*x**2+c*x-r)
    f2 = s*(1-t)*math.cos(x)+s
    result = f1**2 + f2
    if z == "A":
        result *= 10
    if z == "B":
        result *= 0.8
    if result > 10:
        return math.log(result, 10)*10
    else:
        return result
#app.run()

start_time = time.time()
server_address = "http://localhost:5000"

conn = Connection(server_address=server_address)

param_defs = {
    "x": pd.MinMaxNumericParamDef(-5, 10),
    "y": pd.MinMaxNumericParamDef(0, 15)
    #, "z": pd.NominalParamDef(["A", "B", "C"])
}

pd_dict = param_defs_to_dict(param_defs)
optimizer_arguments = {
            "initial_random_runs": 5,
            "acquisition_hyperparams": {},
            "num_gp_restarts": 5,
            #"acquisition": "ExpectedImprovement", #TODO string/gfunction translation?
            "kernel_params": {},
            "kernel": "matern52",
            "mcmc": True,
            "num_precomputed": 5,
            "multiprocessing": "queue"
        }
#print("Looking ofr exps")
#print conn.get_all_experiments()
#print("Finished looking")

number_worker = 1
total_steps = 50

exp_name = "scaled_branin_bay_w%i_no_nominal" %number_worker
#exp_name= "scaled_branin_random"



exp_id = conn.init_experiment(
    name=exp_name,
    param_defs=pd_dict,
    optimizer="BayOpt",
    optimizer_arguments=optimizer_arguments,
    minimization=True,
    blocking=False
)
print("Init exp successful: %s" %exp_id)

print(conn.get_all_experiment_ids())

for i in range(total_steps/number_worker):
    step_time = time.time()
    cands = []
    for j in range(number_worker):
        cands.append(conn.get_next_candidate(exp_id, True, timeout=0))
    for cand in cands:
        cand["result"] = scaled_branin_hoo(**cand["params"])
        print(cand["result"])
        cand["worker_information"] = "Worker info changed."
    for cand in cands:
        conn.update(exp_id, cand, "finished")
    print("Finished %i\t%f" %(i*number_worker, time.time()-step_time))
print("Finished.")
end_time = time.time()
cand = conn.get_best_candidate(exp_id)
print("Best Candidate: %s" %cand)
print("Best result: %s" %cand["result"])
print("Duration: %f" %(end_time-start_time))
#fig, axs = conn.get_figure_results_per_step(exp_name)
#fig.show()