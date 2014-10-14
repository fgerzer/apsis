from abc import ABCMeta, abstractmethod


class ParamDef(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def is_in_parameter_domain(self, value):
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


class NumericParamDef(ParamDef):
    lower_bound = None
    upper_bound = None

    def is_in_parameter_domain(self, value):
        if self.lower_bound <= value <= self.upper_bound:
            return True

        return False

    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

