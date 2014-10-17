from abc import ABCMeta, abstractmethod


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

        self.values = values

    def is_in_parameter_domain(self, value):
        return value in self.values


class OrdinalParamDef(NominalParamDef, ComparableParameterDef):
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
        #integer type of list index
        if self.values.index(one) < self.values.index(two):
            return -1
        if self.values.index(one) > self.values.index(two):
            return 1

        return 0


class NumericParamDef(ParamDef, ComparableParameterDef):
    lower_bound = None
    upper_bound = None

    def is_in_parameter_domain(self, value):
        if self.lower_bound <= value <= self.upper_bound:
            return True

        return False

    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def compare_values(self, one, two):
        """
        Compare values of this numerical data type. Return is the same
        semantic as in __cmp__.
        Comparison takes place based on the numerical values given in the
        arguments 'one' and 'two'. For them the comparison is forwarded
        to their own python __cmp__ method.


        :param one: the first value used in comparison
        :param two: the second value used in comparison
        :return:
            Returns  negative integer if one < two,
            zero if one == two, a positive integer if one > two
        """
        if not (self.is_in_parameter_domain(one) and
                    self.is_in_parameter_domain(two)):
            raise ValueError(
                "Values not comparable! Either one or the other is not in the "
                "values domain")

        # if both values are value forward comparison to their __cmp__
        # thich all numeric types should have
        if one < two:
            return -1
        if one > two:
            return 1

        return 0
