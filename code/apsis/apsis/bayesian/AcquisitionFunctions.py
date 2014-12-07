from abc import ABCMeta, abstractmethod
import numpy as np
import scipy.optimize
import logging
from scipy.stats import multivariate_normal
from apsis.models.ParamInformation import NumericParamDef, NominalParamDef
import random


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

    def compute_proposal(self, args_, refitted=True, number_proposals=1):
        evaluated_params = []
        evaluated_acq = []
        sum_acq = []
        best_param_idx = 0
        best_score = float("inf")
        param_defs = args_['param_defs']

        random_steps = args_.get("random_search_steps", max(1000, number_proposals))

        for i in range(random_steps):
            param_eval = []
            for p in param_defs:
                if isinstance(p, NumericParamDef):
                    param_eval.append(random.random())
                elif isinstance(p, NominalParamDef):
                    param_eval.append(random.choice(p.values))
                else:
                    raise TypeError("Tried using an acquisition function on "
                                    "%s, which is an object of type %s."
                                    "Only NominalParamDef and "
                                    "NumericParamDef are supported."
                                    %(str(p), str(type(p))))
            param_eval = np.array(param_eval)
            score = self.compute_minimizing_evaluate(param_eval, args_)
            if score < best_score:
                best_param_idx = i
                best_score = score
            evaluated_params.append(param_eval)
            evaluated_acq.append(score)
            if len(sum_acq) > 0:
                sum_acq.append(score + sum_acq[-1])
            else:
                sum_acq.append(score)

        proposals = []
        if refitted:
            proposals.append(evaluated_params[best_param_idx])
        while len(proposals) < number_proposals:
            next_prop_idx = 0
            sum_rand = random.uniform(0, sum_acq[-1])
            while sum_rand < sum_acq[next_prop_idx]:
                next_prop_idx += 1
            proposals.append(evaluated_params[next_prop_idx])

        return proposals


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
        if dimensions == 1 and not isinstance(x, list):
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

    def compute_minimizing_evaluate(self, x, args_):
        """
        Changes the sign of the evaluate function.
        """
        value = self.evaluate(x, args_)
        value = -value

        return value