from apsis.models.ParamInformation import NumericParamDef, OrdinalParamDef, \
    NominalParamDef, LowerUpperNumericParamDef
import nose.tools as nt

__author__ = 'ajauch'

class TestNumericParamDef(object):
    """
    Test class to capture tests for all parameter definition objects
    """

    def test_compare_values(self):
        #test numeric param def
        numeric_param = LowerUpperNumericParamDef(0, 5)

        assert numeric_param.compare_values(3.795, 4.20) == -1
        assert numeric_param.compare_values(3.795, 3.795000) == 0
        assert numeric_param.compare_values(4.3, 3.5) == 1

    @nt.raises(ValueError)
    def test_compare_values_exception(self):
        numeric_param = LowerUpperNumericParamDef(0, 5)

        numeric_param.compare_values(7, 1.5)


    def test_initialization_success(self):
        numeric_param = LowerUpperNumericParamDef(4,6)

    #@nt.raises(ValueError)
    def test_initialization_error(self):
        #TODO after change of initialization

        pass


class TestOrdinalParamDef(object):
    def test_compare_values(self):
        #test ordinal param def
        dummy_object = OrdinalParamDef([5, 3, 7, 1])
        assert dummy_object.compare_values(7, 1) == -1
        assert dummy_object.compare_values(7, 7) == 0
        assert dummy_object.compare_values(1, 7) == 1

    @nt.raises(ValueError)
    def test_compare_values_exception(self):
        dummy_object = OrdinalParamDef([5, 3, 7, 1])

        dummy_object.compare_values(8, 1)

    def test_initialization_success(self):
        dummy_object = OrdinalParamDef([5, 3, 7, 1])


    @nt.raises(ValueError)
    def test_initialization_error_no_list(self):
        #try not a real list
        dummy_object = OrdinalParamDef(5)


class TestNominalParamDef(object):

    @nt.raises(ValueError)
    def test_initialization_error_empty_list(self):
        nominal = NominalParamDef([])

    def test_initialization_success(self):
        nominal = NominalParamDef([5, 3, 7, 1])
