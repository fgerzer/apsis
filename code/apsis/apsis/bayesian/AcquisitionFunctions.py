from abc import ABCMeta, abstractmethod
import numpy as np
import scipy.optimize
import logging
from scipy.integrate import quad
from scipy.stats import multivariate_normal

class AcquisitionFunction(object):
    __metaclass__ = ABCMeta

    params = None

    def __init__(self, params=None):
        if params is None:
            self.params = {}

        self.params = params

        pass

    @abstractmethod
    def evaluate(self, x, args_):
        if args_ is None:
            raise ValueError("No arguments dict given!")

        pass

    @abstractmethod
    def compute_max(self, args_):
        if args_ is None:
            raise ValueError("No arguments dict given!")

        pass

class ExpectedImprovement(AcquisitionFunction):
    def __init__(self, params=None):
        super(ExpectedImprovement, self).__init__(params=None)

    def compute_max(self, args_):
        dimensions = len(args_['param_defs'])

        logging.debug("dimensions of param defs %s", str(dimensions))

        initial_guess = [0.5] * dimensions
        initial_guess = tuple(initial_guess)

        bounds = tuple([(0, 1) * dimensions])

        logging.debug("Random guess %s length %s, bounds %s, length %s",
                      str(initial_guess), str(len(initial_guess)), str(bounds),
                      str(len(bounds)))

        # use scipy.optimize.minimize,
        # make sure to use minimizing value from the result object
        minimum = scipy.optimize.minimize(self.compute_negated_evaluate,
                                          initial_guess, args=tuple([args_]),
                                          bounds=bounds, method="SLSQP").x
        #cur = np.zeros((1, 1))
        #minimum = np.zeros((1, 1))
        #min_value = self.evaluate(minimum, args_)[0, 0]
        #for i in range(1000):
        #    cur[0, 0] += 1./1000
        #    if (self.evaluate(cur, args_))[0, 0] > min_value:
        #        #print("new max: %s, %s" %(str(self.evaluate(cur, args_)), str(cur)))
        #        minimum[0, 0] = cur[0, 0]
        #        min_value = self.evaluate(cur, args_)
                #print("new max: %s, %s" %(str(self.evaluate(minimum, args_)), str(minimum)))

        print("min. " + str(minimum))
        return minimum

    def compute_negated_evaluate(self, x, args_):
        value = self.evaluate(x, args_)
        value = -value

        return value



    def evaluate(self, x, args_):
        cur_format = np.zeros((1, 1))
        cur_format[0, 0] = x
        #TODO !!!! Only for testing!!!
        mean, variance, _025pm, _975pm = args_['gp'].predict(cur_format)

        #logging.debug("Evaluating GP mean %s, var %s, _025 %s, _975 %s", str(mean),
        #              str(variance), str(_025pm), str(_975pm))

        # do not standardize on our own, but use the mean, and covariance
        #we get from the gp
        #pdf = scipy.stats.multivariate_normal.pdf
        #cdf_calculate = quad(pdf, 0, x, (mean[0, :], variance))


        #return cdf_calculate[0] * max(0, mean - args_["cur_max"])
        #TODO scale is stand dev, variance is var. Need to scale.
        #return 1 - scipy.stats.norm(loc=mean, scale=variance).cdf(args_["cur_max"])# *max(0, mean - args_["cur_max"])

        Z = (mean - args_["cur_max"])/variance
        cdfZ = 1 - scipy.stats.norm(loc=mean, scale=variance).cdf(Z)# *max(0, mean - args_["cur_max"])
        pdfZ = 1 - scipy.stats.norm(loc=mean, scale=variance).pdf(Z)# *max(0, mean - args_["cur_max"])
        if variance != 0:
            return (mean - args_["cur_max"])*cdfZ + variance*pdfZ
        else:
            return 0



class ProbabilityOfImprovement(AcquisitionFunction):
    def evaluate(self, x, args_):
        # TODO arg check: gpy, x, bestY,

        mean, variance, _025pm, _975pm = args_['gp'].predict(x)

        logging.debug("Evaluating GP mean %s, var %s, _025 %s, _975 %s", str(mean),
                      str(variance), str(_025pm), str(_975pm))

        # do not standardize on our own, but use the mean, and covariance
        #we get from the gp
        #TODO only one dimensional - don't need multivariate
        pdf = scipy.stats.multivariate_normal.pdf
        cdf_calculate = quad(pdf, 0, x, (mean[0, :], variance))

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

        # use scipy.optimize.minimize,
        # make sure to use minimizing value from the result object
        minimum = scipy.optimize.minimize(self.compute_negated_evaluate,
                                          initial_guess, args=tuple([args_]),
                                          bounds=bounds, method="Anneal").x

        return minimum

    def compute_negated_evaluate(self, x, args_):
        value = self.evaluate(x, args_)
        value = -value

        return value