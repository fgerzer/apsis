__author__ = 'Frederik Diehl'

import apsis.models.parameter_definition as pd

def param_defs_to_dict(param_defs):
    """
    Translates paramdefs to a dictionary.

    Parameters
    ----------
    param_defs : dict
        A dictionary with string keys and ParamDef instance values.

    Result
    ------
    param_dict : dict
        Dictionary representing the parameter definitions. Has the
        following format:
        For each parameter, one entry whose key is the name of the
        parameter as a string. The value is a dictionary whose "type" field
        is the name of the ParamDef class, and whose other fields are the
        kwarg fields of that constructor.
    """
    param_dict = {}
    for k in param_defs:
        param_dict[k] = _param_def_to_dict(param_defs[k])
    return param_dict


def _param_def_to_dict(param_def):
    """
    Translates a single parameter definition to a dictionary.

    Parameters
    ----------
    param_def : ParamDef
        A ParamDef subclass instance.

    Result
    ------
    param_dict : dict
        A dictionary whose "type" field
        is the name of the ParamDef class, and whose other fields are the
        kwarg fields of that constructor.
    """
    dict = param_def.to_dict()
    dict["type"] = param_def.__class__.__name__
    return dict


def dict_to_param_defs(dict):
    """
    Translates a dictionary to paramdefs.

    Parameters
    ----------
    dict : dict
        Dictionary representing the parameter definitions. Has the
        following format:
        For each parameter, one entry whose key is the name of the
        parameter as a string. The value is a dictionary whose "type" field
        is the name of the ParamDef class, and whose other fields are the
        kwarg fields of that constructor.

    Result
    ------
    param_defs : dict
        A dictionary with string keys and ParamDef instance values.
    """
    param_defs = {}
    for k in dict:
        param_defs[k] = _dict_to_param_def(dict[k])
    return param_defs


def _dict_to_param_def(param_dict):
    """
    Translates a dictionary to paramdefs.

    Parameters
    ----------
    param_dict : dict
        A dictionary whose "type" field
        is the name of the ParamDef class, and whose other fields are the
        kwarg fields of that constructor.
        
    Result
    ------
    param_def : ParamDef
        A paramDef instance corresponding to the translated param_dict.
    """
    param_type = getattr(pd, param_dict["type"])
    del param_dict["type"]
    return param_type(**param_dict)