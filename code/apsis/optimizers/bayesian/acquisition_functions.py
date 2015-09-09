from abc import ABCMeta, abstractmethod
import numpy as np
import scipy.optimize
import logging
from scipy.stats import multivariate_normal
from apsis.models.parameter_definition import NumericParamDef, PositionParamDef
import random
from apsis.utilities.logging_utils import get_logger

class AcquisitionFunction(object):
    """
    An acquisition function is used to decide which point to evaluate next.

    For a detailed explanation, see for example "A Tutorial on Bayesian
    Optimization of Expensive Cost Functions, with Application to Active User
    Modeling and Hierarchical Reinforcement Learning", Brochu et.al., 2010
    In general, each acquisition function implements two functions, evaluate
    and compute_max.
    """


    _logger = None
    params = None
    LOG_FILE_NAME = "acquisition_functions.log"
    minimizes = True

    def __init__(self, params=None):
        self._logger = get_logger(self, specific_log_name=self.LOG_FILE_NAME)
        if params is None:
            params = {}
        self.params = params

    @abstractmethod
    def evaluate(self, x, gp, experiment):
        """
        Evaluates the gp on the point x.

        Parameters
        ----------
        x : dict
            Dictionary of the parameters on point x
        gp : GPy gp
            The gp on which to evaluate
        experiment : Experiment
            The experiment for further information.

        Returns
        -------
        eval : float
            The value of the acquisition function for the gp on point x.
        """
        pass

    def _compute_minimizing_evaluate(self, x, gp, experiment):
        """
        One problem is that, as a standard, scipy.optimize only searches
        minima. This means we have to convert each acquisition function to
        the minima meaning the best result.
        This the function to do so. Each compute_max can therefore just call
        this function, and know that the returned function has the best value
        as a global minimum.
        Whether to actually return the negative evaluate or not is set by the
        self.minimizes parameter. If true, it will not change the sign.

        Function signature is as evaluate.
        """
        value = self.evaluate(x, gp, experiment)
        if self.minimizes:
            return value
        else:
            return -value

    def compute_proposals(self, gp, experiment, number_proposals=1, return_max=True):
        max_searcher = "none"
        multi_searcher = "none"
        if return_max:
            max_searcher = self.params.get("max_searcher", "random")
            if number_proposals > 1:
                multi_searcher = self.params.get("multi_searcher", "random")
        else:
            multi_searcher = self.params.get("multi_searcher", "random")

        proposals = []

        good_results = []

        if max_searcher != "none":
            max_searcher = getattr(self, "max_searcher_" + max_searcher)
            max_prop, good_results_cur = max_searcher(gp, experiment)
            if good_results_cur is not None:
                good_results.extend(good_results_cur)
            proposals.append(max_prop)
        if multi_searcher != "none":
            multi_searcher = getattr(self, "multi_searcher_" + multi_searcher)
            multi_prop, good_results_cur = multi_searcher(gp, experiment,
                                                          good_results=good_results,
                                                          number_proposals=number_proposals-1)
            if good_results_cur is not None:
                good_results.extend(good_results_cur)
            proposals.extend(multi_prop)

        return proposals

    def max_searcher_random(self, gp, experiment, good_results=None):
        if good_results is None:
            good_results = []
        optimization_random_steps = self.params.get("optimization_random_steps", 1000)

        evaluated_params = []

        best_param_idx = 0
        best_score = float("inf")
        param_defs = experiment.parameter_definitions
        param_names = sorted(param_defs.keys())

        for i in range(optimization_random_steps):
            param_dict_eval = self._compute_random_prop(experiment)
            score = self._compute_minimizing_evaluate(param_dict_eval, gp, experiment)
            if score < best_score:
                best_param_idx = i
                best_score = score
            evaluated_params.append((param_dict_eval, score))

        max_prop = evaluated_params[best_param_idx]
        del evaluated_params[best_param_idx]
        evaluated_params.extend(good_results)
        return max_prop, evaluated_params

    def multi_searcher_random(self, gp, experiment, good_results=None, number_proposals=1):
        if good_results is None:
            good_results = []#TODO

        evaluated_params = []

        optimization_random_steps = self.params.get("optimization_random_steps", 1000)

        random_steps = max(optimization_random_steps, number_proposals) - len(good_results)

        if random_steps > 0:
            for i in range(optimization_random_steps):
                param_dict_eval = self._compute_random_prop(experiment)
                score = self._compute_minimizing_evaluate(param_dict_eval, gp, experiment)
                evaluated_params.append((param_dict_eval, score))

        evaluated_params.extend(good_results)
        evaluated_params.sort(key=lambda prop: prop[1])
        return evaluated_params[:number_proposals], evaluated_params[number_proposals:]


    def _compute_random_prop(self, experiment):
        param_defs = experiment.parameter_definitions
        param_dict_eval = {}
        param_names = sorted(param_defs.keys())
        for pn in param_names:
            pdef = param_defs[pn]
            param_dict_eval[pn] = np.random.uniform(0, 1, pdef.warped_size())
        return param_dict_eval

    def _translate_dict_vector(self, x):
        """
        We translate from a dictionary to a list format for a point's params.

        Parameters
        ----------
        x : dictionary of string keys
            The dictionary defining the point's param values.

        Returns
        -------
        param_to_eval : vector
            Vector of the points' parameter values in order of key.
        """
        param_to_eval = []
        param_names = sorted(x.keys())
        for pn in param_names:
            param_to_eval.extend(x[pn])

        return param_to_eval

    def _translate_vector_dict(self, x_vector, experiment):
        """
        We translate from a vector format to a dictionary of a point's params.

        Parameters
        ----------
        x_vector : vector
            Vector of the points' parameter values. They are assumed to be
             in order of key.

        Returns
        -------
        x : dictionary of string keys
            The dictionary defining the point's param values.
        """
        x_dict = {}

        param_names_sorted = sorted(experiment.parameter_definitions.keys())
        warped_lengths = []
        for pn in param_names_sorted:
            warped_lengths.append(experiment.parameter_definitions[pn].warped_size())
        index = 0
        for i, pn in enumerate(param_names_sorted):
            x_dict[pn] = x_vector[index:index+warped_lengths[i]]
            index += warped_lengths[i]

        return x_dict

    def _translate_vector_nd_array(self, x_vec):
        """
        We translate from a vector of x_vec's params to a numpy nd_array.

        Parameters
        ----------
        x_vec : vector
            Vector of the points' parameter values. They are assumed to be
             in order of key.

        Returns
        -------
        param_nd_array : numpy nd_array
            nd_array of the points' parameter values. They are assumed to be
            in order of key.
        """
        param_nd_array = np.zeros((1, len(x_vec)))
        for i in range(len(x_vec)):
            param_nd_array[0,i] = x_vec[i]

        return param_nd_array

    def in_hypercube(self, x_vec):
        for i in range(len(x_vec)):
            if not 0 <= x_vec[i] <= 1:
                return False
        return True


class GradientAcquisitionFunction(AcquisitionFunction):

    @abstractmethod
    def gradient(self, x, gp, experiment):
        pass

    def _compute_minimizing_gradient(self, x, gp, experiment):
        result = self.gradient(x, gp, experiment)
        if self.minimizes:
            return result
        else:
            return -result

    def max_searcher_LBFGSB(self, gp, experiment, good_results=None):
        bounds = []
        for pd in experiment.parameter_definitions.values():
            bounds.extend([(0.0, 1.0) for x in range(pd.warped_size())])
        if good_results is None:
            good_results = []
        good_results.append(self._compute_random_prop(experiment))

        scipy_optimizer_results = []

        random_restarts = self.params.get("num_restarts", 10)

        for i in range(random_restarts):
            initial_guess = self._translate_dict_vector(self._compute_random_prop(experiment))

            result = scipy.optimize.minimize(self._compute_minimizing_evaluate,
                                                 x0=initial_guess, method="L-BFGS-B",
                                                 jac=self._compute_minimizing_gradient,
                                                 options={'disp': False},
                                                 bounds=bounds,
                                                 args=tuple([gp, experiment]))

            x_min = result.x
            f_min = result.fun
            success = result.success
            if success:
                x_min_dict = self._translate_vector_dict(x_min, experiment)
                if self.in_hypercube(x_min):
                        scipy_optimizer_results.append((x_min_dict, f_min))

        scipy_optimizer_results.extend(good_results)
        best_idx = [x[1] for x in scipy_optimizer_results].index(min([x[1] for x in scipy_optimizer_results]))
        max_prop = scipy_optimizer_results[best_idx]
        del scipy_optimizer_results[best_idx]
        return max_prop, scipy_optimizer_results



class ExpectedImprovement(GradientAcquisitionFunction):
    """
    Implements the Expected Improvement acquisition function.
    See page 13 of "A Tutorial on Bayesian Optimization of Expensive Cost
    Functions, with Application to Active User Modeling and Hierarchical
    Reinforcement Learning", Brochu et. al., 2010.
    """

    minimizes = False

    def _evaluate_vector(self, x_vec, gp, experiment):
        """
        Evaluates the value of the gp at the point x_vec.

        Parameters
        ----------
        x_vec : vector
            The vector defining the point.
        gp : GPy gp
            The gp on which to evaluate
        experiment : experiment
            Some acquisition functions require more information about the
            experiment.

        Results
        -------
        ei_value : vector
            The value of this acquisition funciton on the point.
        ei_gradient : vector
            The value of the gradient on the point
        """
        x_value = self._translate_vector_nd_array(x_vec)

        #mean, variance and their gradients
        mean, variance = gp.predict(x_value)
        gradient_mean, gradient_variance = gp.predictive_gradients(x_value)

        #gpy does everythin in matrices
        gradient_mean = gradient_mean[0]
        #gpy returns variance in row matrices.
        gradient_variance = np.transpose(gradient_variance)

        #these values should be real scalars!
        mean = mean[0][0]
        variance = variance[0][0]

        std_dev = variance ** 0.5

        #Formula adopted from the phd thesis of Jasper Snoek page 48 with
        # \gamma equals Z here
        #Additionally support for the exploration exploitation trade-off
        #as suggested by Brochu et al.
        x_best = experiment.best_candidate.result

        #handle case of maximization
        sign = 1
        if not experiment.minimization_problem:
            sign = -1

        z_numerator = sign * (x_best - mean + self.params.get("exploitation_exploration_tradeoff", 0))

        ei_value = 0
        ei_gradient = 0
        if std_dev != 0:
            z = float(z_numerator) / std_dev

            cdf_z = scipy.stats.norm().cdf(z)
            pdf_z = scipy.stats.norm().pdf(z)

            ei_value = z_numerator * cdf_z + std_dev * pdf_z

            #compute ei gradient
            #new implementation based on own derivation
            ei_gradient_part1 = (1/(2*variance)) * ei_value * gradient_variance
            ei_gradient_part2 = -1 * sign * gradient_mean * cdf_z
            ei_gradient_part3 = -1 * gradient_variance * cdf_z * z * (1/(2*std_dev))
            ei_gradient = ei_gradient_part1 + ei_gradient_part2 + ei_gradient_part3

            ei_gradient = np.transpose(ei_gradient)[0]

        return ei_value, ei_gradient

    def _evaluate_vector_gradient(self, x_vec, gp, experiment):
        """
        Evaluates the gradient of the gp at the point x_vec.

        Parameters
        ----------
        x_vec : vector
            The vector defining the point.
        gp : GPy gp
            The gp on which to evaluate
        experiment : experiment
            Some acquisition functions require more information about the
            experiment.

        Results
        -------
        gradient : vector
            The value of the gradient on the point
        """
        value, grad = self._evaluate_vector(x_vec, gp, experiment)

        return grad

    def gradient(self, x, gp, experiment):
        x_value = self._translate_dict_vector(x)
        value, gradient = self._evaluate_vector(x_value, gp, experiment)
        return gradient

    def evaluate(self, x, gp, experiment):
        x_value = self._translate_dict_vector(x)
        value, gradient = self._evaluate_vector(x_value, gp, experiment)
        return value

class ProbabilityOfImprovement(AcquisitionFunction):
    """
    Implements the probability of improvement function.

    See page 12 of "A Tutorial on Bayesian Optimization of Expensive Cost
    Functions, with Application to Active User Modeling and Hierarchical
    Reinforcement Learning", Brochu et. al., 2010.
    """
    minimizes = False

    def evaluate(self, x, gp, experiment):
        """
        Evaluates the function.
        """
        dimensions = len(experiment.parameter_definitions)
        x_value_vector = self._translate_dict_vector(x)
        x_value = self._translate_vector_nd_array(x_value_vector)

        mean, variance = gp.predict(x_value)

        # do not standardize on our own, but use the mean, and covariance
        # we get from the gp
        stdv = variance ** 0.5
        x_best = experiment.best_candidate.result
        z = (x_best - mean)/stdv

        cdf = scipy.stats.norm().cdf(z)
        result = cdf
        if not experiment.minimization_problem:
            result = 1 - cdf
        return result