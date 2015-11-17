from apsis.utilities.benchmark_functions import branin_func
import sys
from apsis_client.apsis_connection import Connection

server_address = "http://localhost:5000"
conn = None

def single_branin_evaluation_step(conn, exp_id):
    """
    Do a single evaluation on the branin function an all what is necessary
    for it
    1. get the next candidate to evaluate from the assistant.
    2. evaluate branin at this pint
    3. tell the assistant about the new result.

    Parameters
    ----------
    LAss : LabAssistant
        The LabAssistant to use.
    experiment_name : string
        The name of the experiment for this evaluation
    """
    to_eval = conn.get_next_candidate(exp_id)
    result = branin_func(to_eval["params"]["x"], to_eval["params"]["y"])
    to_eval["result"] = result
    conn.update(exp_id, to_eval, "finished")

    return to_eval

def demo_branin(steps=20, random_steps=5, cv=5, disable_auto_plot=False):
    conn = Connection(server_address)


    optimizers = ["RandomSearch", "BayOpt"]
    optimizer_arguments= [{}, {"initial_random_runs": random_steps} ]


    param_defs = {
        "x": {"type": "MinMaxNumericParamDef", "lower_bound": -5, "upper_bound": 10},
        "y": {"type": "MinMaxNumericParamDef", "lower_bound": 0, "upper_bound": 15},
    }

    exp_ids = []
    for i, o in enumerate(optimizers):
        exp_id = conn.init_experiment(o, o, param_defs,
                         minimization=True, optimizer_arguments=optimizer_arguments[i])#{"multiprocessing": "none"})
        exp_ids.append(exp_id)

    print("Initialized all optimizers.")

    for i in range(steps*cv):
        if i > 0 and i%10 == 0:
            print("finished %i" %i)
        for e_id in exp_ids:
            single_branin_evaluation_step(conn, e_id)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        server_address = sys.argv[1]
    print("Connecting to %s" %server_address)
    demo_branin(steps=50, random_steps=10, cv=1)
