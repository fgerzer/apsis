__author__ = 'Frederik Diehl'

from apsis.models.parameter_definition import *
from nose.tools import assert_equal, assert_raises, assert_items_equal, assert_true, assert_false

class TestParameterDefinitions(object):

    def test_nominal_param_def(self):
        with assert_raises(ValueError):
            _ = NominalParamDef(False)
        with assert_raises(ValueError):
            _ = NominalParamDef([])
        test = NominalParamDef(["A", "B", "C"])
        assert_items_equal(test.values, ["A", "B", "C"])

        assert_true(test.is_in_parameter_domain("A"))
        assert_false(test.is_in_parameter_domain(1))

    def test_ordinal_param_def(self):
        with assert_raises(ValueError):
            _ = OrdinalParamDef(False)
        with assert_raises(ValueError):
            _ = OrdinalParamDef([])
        test = OrdinalParamDef(["A", "B", "C"])
        assert_items_equal(test.values, ["A", "B", "C"])

        assert_true(test.is_in_parameter_domain("A"))
        assert_false(test.is_in_parameter_domain(1))
        assert_equal(test.distance("A", "B"), 1./3)
        assert_equal(test.distance("A", "C"), 2./3)
        assert_equal(test.compare_values("A", "B"), -1)
        assert_equal(test.compare_values("A", "A"), 0)

    def test_min_max_def(self):
        with assert_raises(ValueError):
            _ = MinMaxNumericParamDef("Bla", 1)
        with assert_raises(ValueError):
            _ = MinMaxNumericParamDef([], 2)
        test = MinMaxNumericParamDef(-1, 10)

        assert_true(test.is_in_parameter_domain(0.5))
        assert_false(test.is_in_parameter_domain(11))
        assert_equal(test.distance(0, 1), 1./11)
        assert_equal(test.distance(-1, 10), 1)