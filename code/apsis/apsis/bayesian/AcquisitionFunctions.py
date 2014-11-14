from abc import ABCMeta, abstractmethod
import numpy as np
import scipy.optimize
import logging
from scipy.stats import multivariate_normal


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
            Optimization core, but several are available. These include at
            least:
             - The acquisition function
             - The current gp instance
             - The score of the currently best point.
        """
        self.params = params

        if self.params is None:
            self.params = {}

        if 'optimization_strategy' in self.params:
            if self.params['optimization_strategy'] in \
                    self.SUPPORTED_OPTIMIZATION_STRATEGIES:
                self.optimization_strategy = self.params[
                    'optimization_strategy']
            else:
                raise ValueError("You specified the non supported optimization"
                                 "strategy %s for maximizing acquisition.",
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

        This can be toggled via the optimization_strategy keyword in the
        __init__ function's params dict. Supported strategies are in
        SUPPORTED_OPTIMIZATION_STRATEGIES.

        Parameters
        ----------
        args_: dict of string keys
            See AcqusitionFunction.evaluate

        Returns
        -------
        result np.ndarray of floats
            The maximum point for the acquisition function.
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

        Otherwise, method signatures are identical to compute_max.
        """
        dimensions = len(args_['param_defs'])

        bounds = tuple([(0., 1.)] * dimensions)

        grid_points = self.params.get('num_grid_point_per_axis', 1000)

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

        Otherwise, method signatures are identical to compute_max.
        """
        dimensions = len(args_['param_defs'])

        logging.debug("Computing max with grid search method %s for %s "
                      "dimensional problem ", str(dimensions))

        cur = np.zeros((1, 1))
        maximum = np.zeros((1, 1))
        min_value = self.evaluate(maximum, args_)[0, 0]

        for i in range(1000):
            cur[0, 0] += 1. / 1000
            cur_obj = self.evaluate(cur, args_)

            if cur_obj[0, 0] > min_value:
                maximum[0, 0] = cur[0, 0]
                min_value = cur_obj

        logging.debug("EI maximum found at %s", str(maximum))

        return maximum

    def compute_max_scipy_optimize(self, args_):
        """
        Computes the maximum of the acquisition function using the scipy
        optimizer.

        Method signatures are identical to compute_max.
        """
        dimensions = len(args_['param_defs'])

        logging.debug("Computing max with scipy optimize method %s for %s "
                      "dimensional problem ", self.optimization_strategy,
                      str(dimensions))

        # prepare for scipy optimize
        initial_guess = [0.5] * dimensions
        initial_guess = tuple(initial_guess)
        bounds = tuple([(0, 1) * dimensions])

        # make sure to use maximizing value from the result object
        maximum = scipy.optimize.minimize(self.compute_minimizing_evaluate,
                                          initial_guess, args=tuple([args_]),
                                          bounds=bounds,
                                          method=self.optimization_strategy).x

        logging.debug("Acquisition function maximum found at %s", str(maximum))

        return maximum

    def compute_minimizing_evaluate(self, x, args_):
        """
        One problem is that, as a standard, scipy.optimize only searches
        minima. This means we have to convert each acquisition function to
        the minima meaning the best result.
        This the function to do so. Each compute_max can therefore just call
        this function, and know that the returned function has the best value
        as a global minimum.
        As a standard - as here - the function is returned unchanged. If you
        require a negated evaluate function, you have to change this.

        Function signature is as evaluate.
        """
        value = self.evaluate(x, args_)
        return value


class ExpectedImprovement(AcquisitionFunction):
    """
    Implements the Expected Improvement acquisition function.
    See page 13 of "A Tutorial on Bayesian Optimization of Expensive Cost
    Functions, with Application to Active User Modeling and Hierarchical
    Reinforcement Learning", Brochu et. al., 2010.
    """

    exploitation_exploration_tradeoff = 0

    def __init__(self, params=None):
        """
        Initializes the EI instance.

        Parameters: dict of string keys
            Defines behaviour of the function. Includes:
            exploitation_tradeoff: float
                See Brochu, page 14.
            Also see AcquisitionFunction for other parameters.
        """
        super(ExpectedImprovement, self).__init__(params)
        if params is None:
            params = {}
        self.exploitation_exploration_tradeoff = params.get(
            "exploitation_tradeoff", 0)

    def compute_minimizing_evaluate(self, x, args_):
        """
        Changes the sign of the evaluate function.
        """
        value = self.evaluate(x, args_)
        return -value

    def evaluate(self, x, args_):
        """
        Evaluates the Expected Improvement acquisition function.
        """
        dimensions = len(args_['param_defs'])
        x_value = x
        if dimensions == 1:
            x_value = np.zeros((1, 1))
            x_value[0, 0] = x

        mean, variance, _025pm, _975pm = args_['gp'].predict(x_value)

        #See issue #32 on github. using the variance works better than std_dev.
        std_dev = variance ** 0.5

        #Formula adopted from the phd thesis of Jasper Snoek page 48 with
        # \gamma equals Z here

        #Additionally support for the exploration exploitation trade-off
        #as suggested by Brochu et al.
        #Z = (f(x_max) - \mu(x)) / (\sigma(x))
        x_best = args_["cur_max"]

        #handle case of maximization
        sign = 1
        if not args_.get("minimization", True):
            sign = -1

        z_numerator = sign * (x_best - mean +
                              self.exploitation_exploration_tradeoff)

        if variance != 0:
            z = float(z_numerator) / std_dev

            cdf_z = scipy.stats.norm().cdf(z)
            pdf_z = scipy.stats.norm().pdf(z)

            return z_numerator * cdf_z + std_dev * pdf_z
        else:
            return 0


class ProbabilityOfImprovement(AcquisitionFunction):
    """
    Implements the probability of improvement function.

    See page 12 of "A Tutorial on Bayesian Optimization of Expensive Cost
    Functions, with Application to Active User Modeling and Hierarchical
    Reinforcement Learning", Brochu et. al., 2010.
    """

    def evaluate(self, x, args_):
        """
        Evaluates the function.
        """
        mean, variance, _025pm, _975pm = args_['gp'].predict(x)

        logging.debug("Evaluating GP mean %s, var %s, _025 %s, _975 %s",
                      str(mean),
                      str(variance), str(_025pm), str(_975pm))

        # do not standardize on our own, but use the mean, and covariance
        # we get from the gp
        cdf = scipy.stats.norm().cdf(x, mean)
        result = cdf
        if not args_.get("minimization", True):
            result = 1 - cdf
        return result

    def compute_negated_evaluate(self, x, args_):
        """
        Changes the sign of the evaluate function.
        """
        value = self.evaluate(x, args_)
        value = -value

        return value