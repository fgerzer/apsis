from abc import ABCMeta, abstractmethod
import random


"""
Base Classes for Param Definition Classes
"""


class ParamDef(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def is_in_parameter_domain(self, value):
        pass


class ComparableParameterDef(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def compare_values(self, one, two):
        """
        Compare values one and two of this datatype. It has to follow
        the same return semantics as the Python standard __cmp__ methods,
        meaning it returns negative integer if one < two, zero if one == two,
        a positive integer if one > two.

        :param one: the first value used in comparison
        :param two: the second value used in comparison
        :return:
            Returns  negative integer if one < two,
            zero if one == two, a positive integer if one > two
        """
        pass


class NominalParamDef(ParamDef):
    values = None

    def __init__(self, values):
        if not isinstance(values, list):
            raise ValueError(
                "You created a NominalParameterDef object without "
                "specifying the possible values list.")

        if len(values) < 1:
            raise ValueError(
                "You need to specify a list of all possible values for this "
                "data type in order to make it beeing used for your "
                "optimization! The given list was empy: " + str(values)
            )

        self.values = values

    def is_in_parameter_domain(self, value):
        return value in self.values


class OrdinalParamDef(NominalParamDef, ComparableParameterDef):
    def __init__(self, values):
        super(OrdinalParamDef, self).__init__(values)

        # to check comparability execute comparison for two random values
        # from the list
        try:
            self.compare_values(random.choice(self.values),
                                random.choice(self.values))
        except:
            raise ValueError("Creation of a OrdinalParamDef parameter not "
                "possible for the values you specified. There was an error "
                "during comparison. Your values need to be comparable. To "
                "make sure they are the best way is to implement __cmp__ in "
                "your data type. If you can't do this, you need to a non-"
                "comparable param def such as NominalParamDef")

    def compare_values(self, one, two):
        """
        Compare values of this ordinal data type. Return is the same
        semantic as in __cmp__.
        Comparison takes place based on the index
        the given values one and two have in the values list in this object.
        Meaning if this ordinal parameter definition has a values list of
        '[3,5,1,4]', then '5' will be considered smaller than '1' and '1'
        bigger than '5' because the index of '1' in this list is higher than
        the index of '5'.


        :param one: the first value used in comparison
        :param two: the second value used in comparison
        :return:
            Returns  negative integer if one < two,
            zero if one == two, a positive integer if one > two
        """
        if one not in self.values or two not in self.values:
            raise ValueError(
                "Values not comparable! Either one or the other is not in the "
                "values domain")

        # if both values exist in list forward comparision to __cmp__ of
        # integer type of list index
        if self.values.index(one) < self.values.index(two):
            return -1
        if self.values.index(one) > self.values.index(two):
            return 1

        return 0


class NumericParamDef(ParamDef, ComparableParameterDef):
    warping_in = None
    warping_out = None

    def __init__(self, warping_in, warping_out):
        self.warping_in = warping_in
        self.warping_out = warping_out

    #TODO add exception catching for not in warping_out.
    def is_in_parameter_domain(self, value):
        if self.warping_out(0) <= value <= self.warping_out(1):
            return True
        return False

    def warp_in(self, value_in):
        return self.warping_in(value_in)

    def warp_out(self, value_out):
        return self.warping_out(value_out)

    def compare_values(self, one, two):
        if not self.is_in_parameter_domain(one):
            raise ValueError("Parameter one = " + str(one) + " not in value "
                "domain.")
        if not self.is_in_parameter_domain(two):
            raise ValueError("Parameter two = " + str(two) + " not in value "
                "domain.")
        if one < two:
            return -1
        elif one > two:
            return 1
        else:
            return 0


class LowerUpperNumericParamDef(NumericParamDef):

    x_min = None
    x_max = None

    def __init__(self, lower_bound, upper_bound):
        self.x_min = lower_bound
        self.x_max = upper_bound

    def warp_in(self, value_in):
        return (value_in - self.x_min)/(self.x_max-self.x_min)

    def warp_out(self, value_out):
        return value_out*(self.x_max - self.x_min) + self.x_min

    def is_in_parameter_domain(self, value):
        return self.x_min <= value <= self.x_max
