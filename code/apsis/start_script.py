__author__ = 'Frederik Diehl'
import apsis
from apsis.assistants.lab_assistant import LabAssistant
from apsis.webservice.REST_interface import app, lqueue
import apsis.webservice.REST_interface as REST_interface
import time
from apsis.models.parameter_definition import *
import multiprocessing
from multiprocessing import reduction



lab_queue = multiprocessing.Queue()
LAss = LabAssistant(lab_queue)
LAss.start()
print("LASS started.")
time.sleep(0.5)
optimizer = "RandomSearch"
name = "test_init_experiment"
param_defs = {
    "x": MinMaxNumericParamDef(0, 1),
    "y": MinMaxNumericParamDef(0, 1)
}
minimization = False
msg = {
    "action": "init_experiment",
    "name": name,
    "optimizer": optimizer,
    "optimizer_arguments": None,
    "param_defs": param_defs,
    "minimization": minimization
}
lab_queue.put(msg)
time.sleep(0.5)
print("init finished.")

processes = []

print("All workers initialized.")

n_runs = 2
n_workers = 2

start_time = time.time()
for i in range(n_workers):

    p = multiprocessing.Process(target=do_one_learn, args=(lab_queue, n_runs, i))
    processes.append(p)
    p.start()

for p in processes:
    p.join()

man = multiprocessing.Manager()

all_exp_queue = man.Queue()
msg_all = {"action": "get_all_candidates", "exp_name": name, "result_queue": all_exp_queue}
lab_queue.put(msg_all)
cands_finished = all_exp_queue.get()["candidates_finished"]
print("all results finsihed: %s" %[str(x) for x in cands_finished])
print("Total number: %i/%i" %(len(cands_finished), n_runs*n_workers))

best_result_queue = man.Queue()

msg_best = {"action": "get_best_candidate", "exp_name": name, "result_queue": best_result_queue}
lab_queue.put(msg_best)
print("best result: %s" %best_result_queue.get())

print("starting app.")
REST_interface.lqueue = lab_queue
app.run()
print("Started app.")


raw_input("Finished calc, waiting to continue.")

lab_queue.put({"action": "exit"})
print("LASS finished.")
print("Total time: %s" %(time.time()-start_time))
