__author__ = 'Frederik Diehl'
import apsis
from apsis.assistants.lab_assistant import LabAssistant
from apsis.webservice.REST_interface import app, lqueue
import apsis.webservice.REST_interface as REST_interface
import time
from apsis.models.parameter_definition import *
import multiprocessing
from multiprocessing import reduction
from apsis_client.apsis_connection import Connection

from apsis.utilities.param_def_utilities import param_defs_to_dict

import apsis.models.parameter_definition as pd


#from apsis.webservice.REST_interface import app

#app.run()

server_address = "http://localhost:5000"

conn = Connection(server_address=server_address)

param_defs = {
    "x": pd.MinMaxNumericParamDef(0, 1)
}

pd_dict = param_defs_to_dict(param_defs)
print("Looking ofr exps")
print conn.get_all_experiments()
print("Finished looking")
#conn.init_experiment(
#    name="test_exp",
#    param_defs=pd_dict,
#    optimizer="RandomSearch",
#    optimizer_arguments=None,
#    minimization=True
#)