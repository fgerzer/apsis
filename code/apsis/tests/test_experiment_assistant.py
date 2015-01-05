__author__ = 'Frederik Diehl'

from apsis.assistants.experiment_assistant import PrettyExperimentAssistant
from apsis.models.parameter_definition import *
import math
import logging

def func(x, y):
    return x*math.cos(10*y)

param_defs = {
    "x": LowerUpperNumericParamDef(0, 1),
    "y": LowerUpperNumericParamDef(0, 5)
}

logging.basicConfig(level=logging.DEBUG)

BAss = PrettyExperimentAssistant("test", "RandomSearch", param_defs=param_defs)
results = []

for i in range(50):
    to_eval = BAss.get_next_candidate()
    print(to_eval)
    result = func(to_eval.params["x"], to_eval.params["y"])
    results.append(result)
    to_eval.result = result
    BAss.update(to_eval)

print(results)
print(BAss.get_best_candidate())
x, y, z = BAss._best_result_per_step_data()
print("XXXX")
print(x)
print(y)
print(z)
print("ZZZZ")
BAss.plot_result_per_step(plot_at_least=0.5)