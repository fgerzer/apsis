__author__ = 'Frederik Diehl'

from apsis.assistants.experiment_assistant import BasicExperimentAssistant
from apsis.models.parameter_definition import *

def func(x, y):
    return x+y

param_defs = {
    "x": LowerUpperNumericParamDef(0, 1),
    "y": LowerUpperNumericParamDef(0, 1)
}

BAss = BasicExperimentAssistant("RandomSearch", param_defs=param_defs)
results = []

for i in range(20):
    to_eval = BAss.get_next_candidate()
    print(to_eval)
    result = func(to_eval.params["x"], to_eval.params["y"])
    results.append(result)
    to_eval.result = result
    BAss.update(to_eval)

print(results)
print(BAss.get_best_candidate())