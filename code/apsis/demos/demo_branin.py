from apsis.utilities.benchmark_functions import branin_func
from apsis.assistants.lab_assistant import PrettyLabAssistant, ValidationLabAssistant
from apsis.models.parameter_definition import *

import logging

def single_branin_evaluation_step(LAss, experiment_name):
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
    to_eval = LAss.get_next_candidate(experiment_name)
    result = branin_func(to_eval.params["x"], to_eval.params["y"])
    to_eval.result = result
    LAss.update(experiment_name, to_eval)

def demo_branin(steps=100):
    logging.basicConfig(level=logging.DEBUG)

    param_defs = {
        "x": MinMaxNumericParamDef(-5, 10),
        "y": MinMaxNumericParamDef(0, 15)
    }

    LAss = ValidationLabAssistant()
    LAss.init_experiment("Random_Branin", "RandomSearch", param_defs, minimization=True)
    LAss.init_experiment("BayOpt_EI_Branin", "BayOpt", param_defs, minimization=True)

    optimizers = ["Random_Branin", "BayOpt_EI_Branin"]

    #evaluate by step for each mdoel.
    for i in range(steps):
        for optimizer in optimizers:
            single_branin_evaluation_step(LAss, optimizer)

    #plot results comparatively
    # a very detailed plot containing all points
    LAss.plot_result_per_step(optimizers)

    #a plot only showing the evaluation of the best result
    LAss.plot_validation(optimizers)


if __name__ == '__main__':
    demo_branin()