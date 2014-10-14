from abc import ABCMeta, abstractmethod

class Param(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def is_in_parameter_domain(self, value):
        pass


class NominalParam(Param):
    def is_in_parameter_domain(self, value):
        return True


class NumericParam(Param):
    lower_bound = None
    upper_bound = None

    def is_in_parameter_domain(self, value):
        if self.lower_bound <= value <= self.upper_bound:
            return True

        return False

    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

