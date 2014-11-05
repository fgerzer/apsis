from abc import ABCMeta, abstractmethod
from itertools import islice
import numpy as np
import scipy.optimize
import logging
from scipy.integrate import quad
from scipy.stats import multivariate_normal
import collections


class AcquisitionFunction(object):
    """
    An acquisition function is used to decide which point to evaluate next.

    For a detailed explanation, see for example "A Tutorial on Bayesian
    Optimization of Expensive Cost Functions, with Application to Active User
    Modeling and Hierarchical Reinforcement Learning", Brochu et.al., 2010
    In general, each acquisition function implements two functions, evaluate
    and compute_max.
    """
    __metaclass__ = ABCMeta

    SUPPORTED_OPTIMIZATION_STRATEGIES = ["SLSQP", "L-BFGS-B", "TNC",
                                         "grid-simple-1d", "grid-scipy-brute"]

    optimization_strategy = "grid-scipy-brute"

    params = None


    def __init__(self, params=None):
        """
        Initializes an acquisition function.

        Parameters
        ----------
        params: dict, keys are strings
            A dictionary of parameters for the corresponding
            acquisition function. New params must be added in the Bayesian
            Optimization core, but several are available. These include at least:
             - The acquisition function
             - The current gp instance
             - The score of the currently best point.
        """
        self.params = params

        if self.params is None:
            self.params = {}

        if 'optimization_strategy' in self.params:
            if self.params[
                'optimization_strategy'] in self.SUPPORTED_OPTIMIZATION_STRATEGIES:
                self.optimization_strategy = self.params[
                    'optimization_strategy']
            else:
                raise ValueError("You specified the non supported optimization"
                                 "strategy for Expected Improvement %s",
                                 self.params['optimization_strategy'])


    @abstractmethod
    def evaluate(self, x, args_):
        """
        Evaluates the function on one point.

        Parameters
        ----------
        x : np.array of floats
            The point where the acquisition function should be evaluated.
        args_: dict of string keys
            Arguments for the evaluation function.

        Raises
        ------
        ValueError: Iff args_ is None.
        """
        if args_ is None:
            raise ValueError("No arguments dict given!")

        pass

    def compute_max(self, args_):
        """
        Computes the point where the acquisition function is maximized.

        Parameters
        ----------
        args_: dict of string keys
            See AcqusitionFunction.evaluate

        Returns
        -------
        maximum of the acquisition function.
        """
        if args_ is None:
            raise ValueError("No arguments dict given!")

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

            'num_grid_point_per_axis': to say how coarse the grid will be.

        :return: the maximizing hyperparam vectorr of expected improvement as
                 ndarray.
        """
        dimensions = len(args_['param_defs'])

        # prepare for scipy optimize
        #TODO these bounds here seem to be ignored. At least for 1d,
        #scipy.optimize does a lot of extra handling for 1d, so we should try 2d.
        #consider replacing bounds by creating a slice object to manually
        #create the grid
        #bounds = tuple([(0.0, 1.0)] * dimensions)


        grid_points = 1000
        #bounds = tuple([slice(0.0, 1.0, 0.1)]*dimensions)
        bounds = tuple([(0., 1.)] * dimensions)
        #bounds = tuple([(0., 1.)])
        logging.debug("Bounds: %s", str(bounds))

        if 'num_grid_point_per_axis' in self.params:
            grid_points = self.params['num_grid_point_per_axis']

        logging.debug("Computing max with scipy optimize method %s for %s "
                      "dimensional problem using %s points per dimension"
                      " and bounds %s bounds type %s.",
                      self.optimization_strategy,
                      str(dimensions), str(grid_points), str(bounds),
                      str(type(bounds[0])))

        # make sure to use maximizing value from the result object
        maximum, max_value, grid, joust = scipy.optimize.brute(
            self.compute_minimizing_evaluate,
            bounds, Ns=grid_points,
            args=tuple([args_]), full_output=True, disp=True, finish=None)
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
            # walk in grid and check current objective value
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

        # prepare for scipy optimize
        initial_guess = [0.5] * dimensions
        initial_guess = tuple(initial_guess)
        bounds = tuple([(0, 1) * dimensions])

        #logging.debug("Random guess %s length %s, bounds %s, length %s",
        #              str(initial_guess), str(len(initial_guess)), str(bounds),
        #              str(len(bounds)))


        # make sure to use maximizing value from the result object
        maximum = scipy.optimize.minimize(self.compute_minimizing_evaluate,
                                          initial_guess, args=tuple([args_]),
                                          bounds=bounds,
                                          method=self.optimization_strategy).x

        logging.debug("EI maximum found at %s", str(maximum))

        return maximum

    def compute_minimizing_evaluate(self, x, args_):
        value = self.evaluate(x, args_)
        return value




class ExpectedImprovement(AcquisitionFunction):
    """
    Implements the Expected Improvement acquisition function.
    See page 13 of "A Tutorial on Bayesian Optimization of Expensive Cost
    Functions, with Application to Active User Modeling and Hierarchical
    Reinforcement Learning", Brochu et. al., 2010.

    Also implements different optimization strategies.
    """

    exploitation_exploration_tradeoff = 0


    def __init__(self, params=None):
        super(ExpectedImprovement, self).__init__(params)

        self.exploitation_exploration_tradeoff = params.get(
            "exploitation_tradeoff", 0)


    def compute_minimizing_evaluate(self, x, args_):
        value = self.evaluate(x, args_)
        return -value

    def evaluate(self, x, args_):
        dimensions = len(args_['param_defs'])
        x_value = x
        if (dimensions == 1):
            x_value = np.zeros((1, 1))
            x_value[0, 0] = x

        mean, variance, _025pm, _975pm = args_['gp'].predict(x_value)

        std_dev = variance  # **0.5
        # logging.debug("Evaluating GP mean %s, var %s, _025 %s, _975 %s", str(mean),
        #              str(variance), str(_025pm), str(_975pm))


        #Formula adopted from the phd thesis of Jasper Snoek page 48 with
        # \gamma equals Z here

        #Additionally support for the exploration exploitation trade-off
        #as suggested by Brochut et al.

        #Z = (f(x_max) - \mu(x)) / (\sigma(x))
        X_best = args_["cur_max"]

        #handle case of maximization
        sign = 1
        if not args_.get("minimization", True):
            sign = -1

        Z_numerator = sign * (
            X_best - mean + self.exploitation_exploration_tradeoff)
        Z = float(Z_numerator) / std_dev

        #cdf_z = \Phi(Z), pdf_z = \phi(Z)
        cdf_z = scipy.stats.norm().cdf(Z)  # *max(0, mean - args_["cur_max"])
        pdf_z = scipy.stats.norm().pdf(Z)  # *max(0, mean - args_["cur_max"])

        if variance != 0:
            return Z_numerator * cdf_z + std_dev * pdf_z
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
        # we get from the gp
        #TODO only one dimensional - don't need multivariate
        pdf = scipy.stats.multivariate_normal.pdf
        cdf_calculate = quad(pdf, 0, x, (mean[0, :], variance))
        result = cdf_calculate[0]
        if not args_.get("minimization", True):
            result = 1 - result
        return result

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