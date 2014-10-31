from abc import ABCMeta, abstractmethod
import numpy as np
import scipy.optimize
import logging
from scipy.integrate import quad
from scipy.stats import multivariate_normal
import collections


class AcquisitionFunction(object):
    __metaclass__ = ABCMeta

    params = None

    def __init__(self, params=None):
        self.params = params

        if self.params is None:
            self.params = {}


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
    SUPPORTED_OPTIMIZATION_STRATEGIES = ["SLSQP", "L-BFGS-B", "TNC",
                                         "grid-simple-1d", "grid-scipy-brute"]

    optimization_strategy = "grid-scipy-brute"

    def __init__(self, params=None):
        super(ExpectedImprovement, self).__init__(params)

        if 'optimization_strategy' in self.params:
            if self.params[
                'optimization_strategy'] in self.SUPPORTED_OPTIMIZATION_STRATEGIES:
                self.optimization_strategy = self.params[
                    'optimization_strategy']
            else:
                raise ValueError("You specified the non supported optimization"
                                 "strategy for Expected Improvement %s",
                                 self.params['optimization_strategy'])


    def compute_max(self, args_):
        #forward to the corresponding method based by optimization strategy
        if self.optimization_strategy == "grid-simple-1d":
            return self.compute_max_grid_search_1d(args_)

        elif self.optimization_strategy == "SLSQP" \
                or self.optimization_strategy == "L-BFGS-B" \
                or self.optimization_strategy == "TNC":
            return self.compute_max_scipy_optimize(args_)

        elif self.optimization_strategy == "grid-scipy-brute":
            return self.compute_max_scipy_grid_search(args_)

    def compute_max_scipy_grid_search(self, args_):
        """
        Grid search implementation falling back to scipy.optimize.brute.

        It relies on the following arguments in the classes' params hash:

            'num_grid_point_per_axsis': to say how coarse the grid will be.

        :return: the maximizing hyperparam vectorr of expected improvement as
                 ndarray.
        """
        dimensions = len(args_['param_defs'])

        #prepare for scipy optimize
        #TODO these bounds here seem to be ignored. At least for 1d,
        #scipy.optimize does a lot of extra handling for 1d, so we should try 2d.
        #consider replacing bounds by creating a slice object to manually
        #create the grid
        bounds = tuple([(0.0, 1.0) * dimensions])
        grid_points = 1000

        if 'num_grid_point_per_axsis' in self.params:
            grid_points = self.params['num_grid_point_per_axsis']

        logging.debug("Computing max with scipy optimize method %s for %s "
                      "dimensional problem using %s points per dimension"
                      " and bounds %s bounds type %s.",
                      self.optimization_strategy,
                      str(dimensions), str(grid_points), str(bounds),
                      str(type(bounds[0])))

        # make sure to use maximizing value from the result object
        maximum = scipy.optimize.brute(self.compute_negated_evaluate,
                                       bounds, Ns=grid_points,
                                       args=tuple([args_]))[0]

        #a bit hacky, but for 1d check if array, if not make one manually
        if dimensions == 1 and not isinstance(maximum, np.ndarray):
            logging.debug("Converting to array manually.")
            maximum = np.asarray([maximum])

        logging.debug("EI maximum found at %s, data type %s", str(maximum),
                      str(type(maximum)))

        return maximum

    def compute_max_grid_search_1d(self, args_):
        """
        Very simple 1 dimensional grid search implementation. Optimization
        relies on that input is 1d and in value range between 0 and 1.

        :param args_:
        :return:
        """
        dimensions = len(args_['param_defs'])

        logging.debug("Computing max with grid search method %s for %s "
                      "dimensional problem ", str(dimensions))

        cur = np.zeros((1, 1))
        maximum = np.zeros((1, 1))
        min_value = self.evaluate(maximum, args_)[0, 0]

        for i in range(1000):
            #walk in grid and check current objective value
            cur[0, 0] += 1. / 1000
            cur_obj = self.evaluate(cur, args_)

            if cur_obj[0, 0] > min_value:
                maximum[0, 0] = cur[0, 0]
                min_value = cur_obj

        logging.debug("EI maximum found at %s", str(maximum))

        return maximum


    def compute_max_scipy_optimize(self, args_):
        dimensions = len(args_['param_defs'])

        logging.debug("Computing max with scipy optimize method %s for %s "
                      "dimensional problem ", self.optimization_strategy,
                      str(dimensions))

        #prepare for scipy optimize
        initial_guess = [0.5] * dimensions
        initial_guess = tuple(initial_guess)
        bounds = tuple([(0, 1) * dimensions])

        #logging.debug("Random guess %s length %s, bounds %s, length %s",
        #              str(initial_guess), str(len(initial_guess)), str(bounds),
        #              str(len(bounds)))


        # make sure to use maximizing value from the result object
        maximum = scipy.optimize.minimize(self.compute_negated_evaluate,
                                          initial_guess, args=tuple([args_]),
                                          bounds=bounds,
                                          method=self.optimization_strategy).x

        logging.debug("EI maximum found at %s", str(maximum))

        return maximum

    def compute_negated_evaluate(self, x, args_):
        value = self.evaluate(x, args_)
        value = -value

        return value


    def evaluate(self, x, args_):
        #TODO @Frederik: What is the cur_format here for???

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

        Z = (mean - args_["cur_max"]) / variance
        cdfZ = 1 - scipy.stats.norm(loc=mean, scale=variance).cdf(
            Z)  # *max(0, mean - args_["cur_max"])
        pdfZ = 1 - scipy.stats.norm(loc=mean, scale=variance).pdf(
            Z)  # *max(0, mean - args_["cur_max"])

        if variance != 0:
            return (mean - args_["cur_max"]) * cdfZ + variance * pdfZ
        else:
            return 0


class ProbabilityOfImprovement(AcquisitionFunction):
    def evaluate(self, x, args_):
        # TODO arg check: gpy, x, bestY,

        mean, variance, _025pm, _975pm = args_['gp'].predict(x)

        logging.debug("Evaluating GP mean %s, var %s, _025 %s, _975 %s",
                      str(mean),
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
        minimum = scipy.optimize.minimize_scalar(self.compute_negated_evaluate,
                                                 initial_guess,
                                                 args=tuple([args_]),
                                                 bounds=bounds,
                                                 method="Anneal").x

        return minimum

    def compute_negated_evaluate(self, x, args_):
        value = self.evaluate(x, args_)
        value = -value

        return value