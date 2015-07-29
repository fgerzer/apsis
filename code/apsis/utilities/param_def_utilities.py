__author__ = 'Frederik Diehl'

from apsis.models.parameter_definition import *

AVAILABLE_PARAM_DEFS = {
    "nominal_param_def": NominalParamDef,
    "ordinal_param_def": OrdinalParamDef,
    "numeric_param_def": NumericParamDef,
    "min_max_numeric_param_def": MinMaxNumericParamDef,
    "position_param_def": PositionParamDef,
    "fixed_value_param_def": FixedValueParamDef,
    "anymptotic_numeric_param_def": AsymptoticNumericParamDef
}

def convert_param_defs(param_def_dict_raw):
    param_def_dict = {}
    for p in param_def_dict:
        param_def_dict[p] = convert_param_def(param_def_dict_raw[p])
    return param_def_dict

def convert_param_def(param_def_dict):
    type = param_def_dict["type"]
    del param_def_dict["type"]
    class_type = AVAILABLE_PARAM_DEFS[type]
    return class_type(**param_def_dict)