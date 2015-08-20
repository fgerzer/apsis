__author__ = 'Frederik Diehl'

from apsis.utilities.param_def_utilities import param_defs_to_dict, dict_to_param_defs, _param_def_to_dict, _dict_to_param_def
from apsis.models.parameter_definition import *

class TestParamDefConverter(object):

    def test_nominal(self):
        pd = NominalParamDef(["a", "b", "c"])
        dict = _param_def_to_dict(pd)
        print(dict)
        new_pd = _dict_to_param_def(dict)
        assert type(pd) == type(new_pd)
        assert pd.values == new_pd.values

    def test_ordinal(self):
        pd = OrdinalParamDef(["a", "b", "c"])
        dict = _param_def_to_dict(pd)
        print(dict)
        new_pd = _dict_to_param_def(dict)
        assert type(pd) == type(new_pd)
        assert pd.values == new_pd.values

    def test_minmax_numeric(self):
        pd = MinMaxNumericParamDef(-2, 20)
        dict = _param_def_to_dict(pd)
        print(dict)
        new_pd = _dict_to_param_def(dict)
        assert type(pd) == type(new_pd)
        assert pd.lower_bound == new_pd.lower_bound
        assert pd.upper_bound == new_pd.upper_bound

    def test_position(self):
        pd = PositionParamDef(["a", "b", "c"], [0, 1, 2])
        dict = _param_def_to_dict(pd)
        print(dict)
        new_pd = _dict_to_param_def(dict)
        assert type(pd) == type(new_pd)
        assert pd.values == new_pd.values
        assert pd.positions== new_pd.positions


    def test_fixed_value(self):
        pd = FixedValueParamDef([0, 1, 2])
        dict = _param_def_to_dict(pd)
        print(dict)
        new_pd = _dict_to_param_def(dict)
        assert type(pd) == type(new_pd)
        assert pd.values == new_pd.values
        assert pd.positions== new_pd.positions


    def test_asymptotic(self):
        pd = AsymptoticNumericParamDef(2, 10)
        dict = _param_def_to_dict(pd)
        print(dict)
        new_pd = _dict_to_param_def(dict)
        assert type(pd) == type(new_pd)
        assert pd.asymptotic_border == new_pd.asymptotic_border
        assert pd.border == new_pd.border