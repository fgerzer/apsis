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

#app.run()

start_time = time.time()
server_address = "http://localhost:5000"

conn = Connection(server_address=server_address)

param_defs = {
    "x": pd.MinMaxNumericParamDef(0, 1),
    "y": pd.MinMaxNumericParamDef(0, 1),
    "z": pd.MinMaxNumericParamDef(0, 1)
}

pd_dict = param_defs_to_dict(param_defs)
#print("Looking ofr exps")
#print conn.get_all_experiments()
#print("Finished looking")
conn.init_experiment(
    name="test_exp",
    param_defs=pd_dict,
    optimizer="RandomSearch",
    optimizer_arguments=None,
    minimization=True
)

print(conn.get_all_experiment_names())

for i in range(10):
    cand = None
    cand = conn.get_next_candidate("test_exp", True, timeout=0)
    print(cand)
    cand["result"] = (cand["params"]["z"]+1)*\
                     (cand["params"]["x"] ** 2 + (cand["params"]["y"] - 1) ** 2)
    time.sleep(1)
    cand["worker_information"] = "Worker info changed."
    conn.update("test_exp", cand, "finished")

    print("Best: %s" %conn.get_best_candidate("test_exp"))
    print("All: %s" %conn.get_all_candidates("test_exp"))
    print("\n\n\n")
print("Finished.")
end_time = time.time()
print("Duration: %f" %(end_time-start_time))