__author__ = 'Frederik Diehl'

from abc import ABCMeta, abstractmethod
import numpy as np
import scipy.optimize
import logging
from scipy.stats import multivariate_normal
from apsis.models.parameter_definition import NumericParamDef, PositionParamDef
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

    logger = None

    params = None

    def __init__(self, params=None):
        self.logger = logging.getLogger(__name__)
        self.params = params

        if self.params is None:
            self.params = {}

    @abstractmethod
    def evaluate(self, x, gp, experiment):
        pass

    def _compute_minimizing_evaluate(self, x, gp, experiment):
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
        value = self.evaluate(x, gp, experiment)
        return value

    def compute_proposals(self, gp, experiment, number_proposals=1,
                          random_steps=1000):
        evaluated_params = []
        evaluated_acq_scores = []
        sum_acq = []

        best_param_idx = 0
        best_score = float("inf")
        param_defs = experiment.parameter_definitions

        param_names = sorted(param_defs.keys())

        random_steps = max(random_steps, number_proposals)

        for i in range(random_steps):
            param_dict_eval = {}
            for pn in param_names:
                pdef = param_defs[pn]
                if isinstance(pdef, NumericParamDef) \
                        or isinstance(pdef, PositionParamDef):
                    param_dict_eval[pn] = random.random()
                else:
                    raise TypeError("Tried using an acquisition function on "
                                    "%s, which is an object of type %s."
                                    "Only "
                                    "NumericParamDef are supported."
                                    %(str(pdef), str(type(pdef))))

            score = self._compute_minimizing_evaluate(param_dict_eval, gp, experiment)

            if score < best_score:
                best_param_idx = i
                best_score = score
            evaluated_params.append(param_dict_eval)
            evaluated_acq_scores.append(score)
            if len(sum_acq) > 0:
                sum_acq.append(score + sum_acq[-1])
            else:
                sum_acq.append(score)

        proposals = []
        proposals.append(evaluated_params[best_param_idx])
        while len(proposals) < number_proposals:
            next_prop_idx = 0
            sum_rand = random.uniform(0, sum_acq[-1])
            while sum_rand < sum_acq[next_prop_idx]:
                next_prop_idx += 1
            proposals.append(evaluated_params[next_prop_idx])
        return proposals

    def _translate_dict_vector(self, x):
        #here we translate from dict to list format for points.
        param_to_eval = []
        param_names = sorted(x.keys())
        for pn in param_names:
            param_to_eval.append(x[pn])

            #And to np.array for gpy.
        param_nd_array = np.zeros((1, len(param_to_eval)))
        #param_nd_array = np.ndarray(param_to_eval)
        for i in range(len(param_to_eval)):
            param_nd_array[0, i] = param_to_eval[i]

        return param_nd_array

class ExpectedImprovement(AcquisitionFunction):
    """
    Implements the Expected Improvement acquisition function.
    See page 13 of "A Tutorial on Bayesian Optimization of Expensive Cost
    Functions, with Application to Active User Modeling and Hierarchical
    Reinforcement Learning", Brochu et. al., 2010.
    """
    exploitation_exploration_tradeoff = 0


    def __init__(self, params = None):
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

    def compute_minimizing_evaluate(self, x, gp, experiment):
        """
        Changes the sign of the evaluate function.
        """
        value = self.evaluate(self, x, gp, experiment)
        return -value



    def evaluate(self, x, gp, experiment):
        """
        Evaluates the Expected Improvement acquisition function.
        """
        dimensions = len(experiment.parameter_definitions)
        x_value = self._translate_dict_vector(x)

        mean, variance = gp.predict(x_value)

        #See issue #32 on github. using the variance works better than std_dev.
        std_dev = variance ** 0.5

        #Formula adopted from the phd thesis of Jasper Snoek page 48 with
        # \gamma equals Z here

        #Additionally support for the exploration exploitation trade-off
        #as suggested by Brochu et al.
        #Z = (f(x_max) - \mu(x)) / (\sigma(x))
        x_best = experiment.best_candidate.result

        #handle case of maximization
        sign = 1
        if not experiment.minimization_problem:
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

    def evaluate(self, x, gp, experiment):
        """
        Evaluates the function.
        """

        dimensions = len(experiment.parameter_definitions)
        x_value = self._translate_dict_vector(x)

        mean, variance = gp.predict(x_value)

        # do not standardize on our own, but use the mean, and covariance
        # we get from the gp
        cdf = scipy.stats.norm().cdf(x, mean)
        result = cdf
        if not experiment.minimization_problem:
            result = 1 - cdf
        return result

    def compute_minimizing_evaluate(self, x, gp, experiment):
        """
        Changes the sign of the evaluate function.
        """
        value = self.evaluate(x, gp, experiment=experiment)
        value = -value

        return value