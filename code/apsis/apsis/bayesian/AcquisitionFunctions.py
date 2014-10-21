from abc import ABCMeta, abstractmethod
import numpy as np
import scipy.optimize
import logging
from scipy.integrate import quad


class AcquisitionFunction(object):
    __metaclass__ = ABCMeta

    params = None

    def __init__(self, params=None):
        if params is None:
            self.params = {}

        self.params = params

        pass

    @abstractmethod
    def evaluate(self, args):
        if args is None:
            raise ValueError("No arguments dict given!")

        pass

    @abstractmethod
    def compute_max(self, args):
        if args is None:
            raise ValueError("No arguments dict given!")

        pass


class ProbabilityOfImprovement(AcquisitionFunction):
    def evaluate(self, x, args_):
        # TODO arg check: gpy, x, bestY,

        mean, variance, _025pm, _975pm = args_['gp'].predict(x)

        logging.debug("GP mean %s, var %s, _025 %s, _975 %s", str(mean),
                      str(variance), str(_025pm), str(_975pm))

        # do not standardize on our own, but use the mean, and covariance
        #we get from the gp
        cdf_calculate = quad(scipy.stats.multivariate_normal.pdf, 0, x,
                             (mean[0, :], variance))

        return cdf_calculate[0]

    def compute_max(self, args_):
        dimensions = len(args_['param_defs'])

        logging.debug("dimensions of param defs %s", str(dimensions))

        initial_guess = [0.5] * dimensions
        initial_guess = tuple(initial_guess)

        bounds = tuple([(0, 1) * dimensions])

        logging.debug("Random guess %s length %s, bounds %s, length %s",
                      str(initial_guess), str(len(initial_guess)), str(bounds),
                      str(len(bounds)))

        """
        use scipy.optimize.minimize,
        make sure to use minimizing value from the result object
        """
        minimum = scipy.optimize.minimize(self.compute_negated_evaluate,
                                          initial_guess, args=tuple([args_]),
                                          method=None, bounds=bounds).x

        return minimum

    def compute_negated_evaluate(self, x, args_):
        value = self.evaluate(x, args_)
        value = -value

        return value