from apsis.utilities.benchmark_functions import branin_func
from apsis.assistants.lab_assistant import PrettyLabAssistant, ValidationLabAssistant
from apsis.models.parameter_definition import *
from apsis.utilities.randomization import check_random_state
import logging
from apsis.utilities.logging_utils import get_logger

logger = get_logger("demos.demo_branin")

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

    return to_eval

def demo_branin(steps=50, random_steps=10, cv=5):
    logging.basicConfig(level=logging.DEBUG)

    #produce the same random state
    random_state_rs = check_random_state(42)

    param_defs = {
        "x": MinMaxNumericParamDef(-5, 10),
        "y": MinMaxNumericParamDef(0, 15)
    }

    LAss = ValidationLabAssistant(cv=cv)
    LAss.init_experiment("Random_Branin", "RandomSearch", param_defs, minimization=True, optimizer_arguments={"random_state": random_state_rs})
    #LAss.init_experiment("BayOpt_EI_Branin", "BayOpt", param_defs, minimization=True, optimizer_arguments={"initial_random_runs": random_steps})

    optimizers = ["Random_Branin", "BayOpt_EI_Branin"]
    optimizer_arguments= [{"random_state": random_state_rs}, {"initial_random_runs": random_steps} ]

    #evaluate random search for 10 steps use these steps as init value for bayesian
    for i in range(random_steps*cv):
        evaluated_candidate = single_branin_evaluation_step(LAss, 'Random_Branin')

    #now clone experiment for each optimizer
    for j in range(1, len(optimizers)):
        LAss.clone_experiments_by_name(exp_name=optimizers[0], new_exp_name=optimizers[j], optimizer="BayOpt", optimizer_arguments=optimizer_arguments[j])
    logger.info("Random Initialization Phase Finished.")
    logger.info("Competitive Evaluation Phase starts now.")

    #from there on go step by step all models
    for i in range(random_steps*cv, steps*cv):
        for optimizer in optimizers:
            single_branin_evaluation_step(LAss, optimizer)

    #plot results comparatively
    # a very detailed plot containing all points
    LAss.plot_result_per_step(optimizers)

    #a plot only showing the evaluation of the best result
    LAss.plot_validation(optimizers)


if __name__ == '__main__':
    demo_branin(steps=50, random_steps=10, cv=10)