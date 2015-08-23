__author__ = 'Frederik Diehl'

import apsis.models.parameter_definition as pd

def param_defs_to_dict(param_defs):
    param_dict = {}
    for k in param_defs:
        param_dict[k] = _param_def_to_dict(param_defs[k])
    return param_dict

def _param_def_to_dict(param_def):
    dict = param_def.to_dict()
    dict["type"] = param_def.__class__.__name__
    return dict

def dict_to_param_defs(dict):
    param_defs = {}
    for k in dict:
        param_defs[k] = _dict_to_param_def(dict[k])
    return param_defs

def _dict_to_param_def(dict):
    param_type = getattr(pd, dict["type"])
    del dict["type"]
    return param_type(**dict)