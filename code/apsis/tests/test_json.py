__author__ = 'Frederik Diehl'

import json
from apsis.models.experiment import Experiment
from apsis.models.parameter_definition import *
from apsis.models.candidate import Candidate, from_dict
from apsis.utilities.json_utils import *

if __name__ == '__main__':

    param_defs = {
        "test_param": MinMaxNumericParamDef(0, 1)
    }
    e = Experiment("test", param_defs)

    print("+++++++++++++++++++++++++++++++++++++++++++++++++")

    c = Candidate({"test_param": 1})
    c.result = 10
    e.add_finished(c)


    print(c)
    print("-----------------------")
    c_dict = c.to_dict()

    print(c_dict)

    print("-----------------------")

    d = from_dict(c_dict)
    print(d)