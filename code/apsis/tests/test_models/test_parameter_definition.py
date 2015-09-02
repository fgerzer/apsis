__author__ = 'Frederik Diehl'

from apsis.models.parameter_definition import *
from nose.tools import assert_equal, assert_raises, assert_items_equal, \
    assert_true, assert_false, assert_almost_equal, assert_less_equal, \
    assert_greater_equal

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


    def test_numeric_def(self):
        f_in = lambda x: float(x)/10
        f_out = lambda x: float(x)*10
        test = NumericParamDef(f_in, f_out)

        assert_true(test.is_in_parameter_domain(0.5))
        assert_false(test.is_in_parameter_domain(11))
        assert_equal(test.distance(0, 1), 0.1)
        assert_equal(test.distance(0, 10), 1)
        with assert_raises(ValueError):
            test.distance("A", 1)
        with assert_raises(ValueError):
            test.distance(0, "B")
        assert_equal(test.compare_values(0, 10), -1)
        assert_equal(test.compare_values(1, 0), 1)

    def test_fixed_def(self):
        test = FixedValueParamDef([0, 1, 2, 3])

        assert_true(test.is_in_parameter_domain(1))
        assert_false(test.is_in_parameter_domain(1.5))
        assert_equal(test.distance(0, 1), 1)
        assert_equal(test.distance(0, 3), 3)
        with assert_raises(ValueError):
            test.distance("A", 1)
        with assert_raises(ValueError):
            test.distance(0, 4)
        assert_equal(test.compare_values(0, 3), -1)
        assert_equal(test.compare_values(1, 0), 1)

    def test_asymptotic_def(self):
        asymptotic = 0
        border = 1
        pd = AsymptoticNumericParamDef(asymptotic, border)
        for i in range(0, 100):
            #x = float(i)/100 * asymptotic + (1-float(i)/100)*border
            x = asymptotic + float(i)/100 * border
            w_i = pd.warp_in(x)
            w_o = pd.warp_out(w_i)
            assert_less_equal(w_i[0], 1)
            assert_greater_equal(w_i[0], 0)
            assert_almost_equal(w_o, min(max(x, 0), 1))