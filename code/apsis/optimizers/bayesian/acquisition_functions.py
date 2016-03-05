from abc import ABCMeta, abstractmethod
import numpy as np
import scipy.optimize
from scipy.stats import multivariate_normal
import random
from apsis.utilities.logging_utils import get_logger


class AcquisitionFunction(object):
    """
    An acquisition function is used to decide which point to evaluate next.

    For a detailed explanation, see for example "A Tutorial on Bayesian
    Optimization of Expensive Cost Functions, with Application to Active User
    Modeling and Hierarchical Reinforcement Learning", Brochu et.al., 2010

    Internally, each AcquisitionFunction implements a couple of max_searcher
    and multi_searcher functions. These work as follows:
    A max_searcher function takes the gp, experiment and (optionally) a
    good_proposals list. It then uses these to compute a proposal maximizing
    the acquisition function, and returns a tuple of this and its score.
    A multi_searcher function takes the gp, experiment, an (optional)
    good_proposals list and a maximum number of proposals. It then uses these
    to return several proposals in a list.
    Additionally, both max_searcher and multi_searcher functions have to return
    an own good_proposals list as a second return value (or None). These are
    evaluated proposals which are not in the first list, and are used to
    reuse computation.
    """

    _logger = None
    params = None
    minimizes = True

    default_max_searcher = "random"
    default_multi_searcher = "random_weighted"

    def __init__(self, params=None):
        """
        Initializes the acquisition function.

        Parameters
        ----------
        params : dict or None, optional
            The dictionary of parameters defining the behaviour of the
            acquisition function. Supports at least max_searcher and
            multi_searcher.
        """
        self._logger = get_logger(self)
        self._logger.debug("Initializing acquisition function. params is %s",
                           params)
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
        # Warning: Logs very often if activated.
        self._logger.log(5, "Computing minimizing evaluate. x is %s, gp is %s,"
                           "experiment is %s", x, gp, experiment)
        value = self.evaluate(x, gp, experiment)
        if self.minimizes:
            # Warning: Logs very often if activated.
            self._logger.log(5, "Is minimizing, returning %s", value)
            return value
        else:
            # Warning: Logs very often if activated.
            self._logger.log(5, "Is maximizing, returning %s", -value)
            return -value

    def compute_proposals(self, gp, experiment, number_proposals=1,
                          return_max=True):
        """
        Computes up to number_proposals proposals.

        If return_max, the first entry in the returned list will be the
        proposal maximizing the acquisition function.

        Parameters
        ----------
        gp : GPy.gp
            The gp.

        experiment : experiment
            The current state of the experiment.

        number_proposals : int, optional
            The maximum number of proposals returned. The acquisition function
             will try its best to return that many proposals, but this cannot
             be guaranteed. By default, returns one proposal.

        return_max : bool, optional
            Whether to try and find the maximum of the proposal. If true (the
            default) will do another evaluation step).

        Returns
        -------
        proposals : list of tuples
            A list of tuples. The first entry of each is a dictionary with the
            parameter name and a 0-1 hypercube entry representing the warped
            parameter value. The second entry is the acquisition function score
            at that point. May return an empty list.
        """
        self._logger.debug("Computing proposals. gp is %s, experiment is %s, "
                           "number_proposals %s, return_max %s",
                           gp, experiment, number_proposals, return_max)
        max_searcher = "none"
        multi_searcher = "none"
        if return_max:
            max_searcher = self.params.get("max_searcher",
                                           self.default_max_searcher)
            self._logger.debug("Returning maximum. Max_searcher is %s",
                               max_searcher)
            if number_proposals > 1:
                multi_searcher = self.params.get("multi_searcher",
                                                 self.default_multi_searcher)
                self._logger.debug("Also returning multiple results; "
                                   "multi_searcher is %s", multi_searcher)
        else:
            multi_searcher = self.params.get("multi_searcher",
                                             self.default_multi_searcher)
            self._logger.debug("Not returning max. multi_searcher is %s",
                               multi_searcher)

        proposals = []

        good_results = []

        if max_searcher != "none":
            self._logger.debug("Beginning to look for max.")
            max_searcher = getattr(self, "max_searcher_" + max_searcher)
            self._logger.debug("Constructed max_search function. max_searcher "
                               "is %s", max_searcher)
            max_prop, good_results_cur = max_searcher(gp, experiment)
            self._logger.debug("Finished search. max_prop is %s", max_prop)
            self._logger.log(5, "good results cur is %s", good_results_cur)
            if good_results_cur is not None:
                good_results.extend(good_results_cur)
                self._logger.log(5, "Extended known good results. Now %s",
                                   good_results)
            proposals.append(max_prop)
            self._logger.log(5, "Appended %s to max_prop, now %s", max_prop,
                               proposals)
        if multi_searcher != "none":
            self._logger.debug("Starting multi_search.")
            multi_searcher = getattr(self, "multi_searcher_" + multi_searcher)
            self._logger.debug("multi_searcher function generated as %s",
                               multi_searcher)
            multi_prop, good_results_cur = multi_searcher(gp, experiment,
                                          good_results=good_results,
                                          number_proposals=number_proposals-1)
            self._logger.debug("Finished multi search. Multi_prop is %s",
                               multi_prop)
            self._logger.log(5, "good_results_cur is %s", good_results_cur)
            if good_results_cur is not None:
                good_results.extend(good_results_cur)
                self._logger.log(5, "Appended (extend) %s to good_results. Now"
                                   " %s", good_results_cur, good_results)
            proposals.extend(multi_prop)
            self._logger.debug("Appended %s to proposals; now %s",
                               multi_prop, proposals)
        self._logger.debug("Returning proposals %s", proposals)
        return proposals

    def max_searcher_random(self, gp, experiment, good_results=None):
        """
        Randomly searches the best result.

        Uses optimization_random_steps in self.params, with a default of 1000.

        For signature details see the introduction in the class docs.
        """
        self._logger.debug("Starting max_searcher_random. gp is %s, experiment"
                           " %s, good_results %s", gp, experiment,
                           good_results)
        if good_results is None:
            good_results = []
        optimization_random_steps = self.params.get("optimization_random_steps"
                                                    , 1000)
        self._logger.debug("Will generated %s random initial steps",
                           optimization_random_steps)
        evaluated_params = []

        best_param_idx = 0
        best_score = float("inf")

        for i in range(optimization_random_steps):
            param_dict_eval = self._gen_random_prop(experiment)
            score = self._compute_minimizing_evaluate(param_dict_eval, gp,
                                                      experiment)
            if score < best_score:
                best_param_idx = i
                best_score = score
            evaluated_params.append((param_dict_eval, score))
        self._logger.debug("Evaluated all steps: %s", evaluated_params)
        max_prop = evaluated_params[best_param_idx]
        del evaluated_params[best_param_idx]
        evaluated_params.extend(good_results)
        self._logger.log(5, "Will return %s and %s", max_prop, evaluated_params)
        return max_prop, evaluated_params

    def multi_searcher_random_best(self, gp, experiment, good_results=None,
                                   number_proposals=1):
        """
        Randomly evaluates a number of proposals, returning the
        number_proposals best.

        Uses optimization_random_steps in self.params, with a default of 1000.

        For signature details see the introduction in the class docs.
        """
        evaluated_params = self._multi_random_ordered(gp, experiment,
                                                      good_results,
                                                      number_proposals)
        self._logger.debug("Evaluated the best multi candidates.")
        self._logger.log(5, "Result is %s", evaluated_params)
        return evaluated_params[:number_proposals], \
               evaluated_params[number_proposals:]


    def multi_searcher_random_weighted(self, gp, experiment,
                                       good_results=None, number_proposals=1):
        """
        Returns number_proposals proposals randomly weighted by their
        acquisition result.

        Uses optimization_random_steps in self.params, with a default of 1000.

        For signature details see the introduction in the class docs.
        """
        self._logger.debug("Beginning random-weighted search. gp is %s,"
                           "experiment is %s, good_results %s, "
                           "number_proposals %s",
                           gp, experiment, good_results, number_proposals)
        evaluated_params = self._multi_random_ordered(gp, experiment,
                                                      good_results,
                                                      number_proposals)
        self._logger.log(5, "Got initial random results: %s", evaluated_params)
        acq_sum = 0
        for p in evaluated_params:
            acq_sum += p[1]
        self._logger.log(5, "Total acquisition function values are %s",
                         acq_sum)
        props = []
        for i in range(number_proposals):
            rand_acq = random.random() * acq_sum
            cur_sum = 0
            for j, p in enumerate(evaluated_params):
                if cur_sum + p[1] > rand_acq:
                    props.append(p)
                    del evaluated_params[j]
                    break
                cur_sum += p[1]
        self._logger.log(5, "Got final results. Props: %s, evaluated_params: "
                           "%s", props, evaluated_params)
        return props, evaluated_params


    def _multi_random_ordered(self, gp, experiment, good_results=None,
                              number_proposals=1):
        """
        Generates a number of random proposals, and returns them ordered. Used
        for other functions.

        Uses optimization_random_steps in self.params, with a default of 1000.
        """
        self._logger.debug("Started multi_random_ordered. gp is %s, "
                           "experiment %s, good_results %s, "
                           "number_proposals %s", gp, experiment,
                           good_results, number_proposals)
        if good_results is None:
            good_results = []

        evaluated_params = []

        optimization_random_steps = self.params.get(
            "optimization_random_steps", 1000)
        self._logger.debug("Set optimization_random_steps to %s",
                           optimization_random_steps)

        random_steps = max(optimization_random_steps, number_proposals) - \
                       len(good_results)
        self._logger.debug("Requires %s random_steps", random_steps)
        if random_steps > 0:
            for i in range(optimization_random_steps):
                param_dict_eval = self._gen_random_prop(experiment)
                score = self._compute_minimizing_evaluate(param_dict_eval, gp,
                                                          experiment)
                evaluated_params.append((param_dict_eval, score))

        evaluated_params.extend(good_results)
        evaluated_params.sort(key=lambda prop: prop[1])
        # Only activate this logger if crazy. Output is huge.
        self._logger.log(5, "Returning %s", evaluated_params)
        return evaluated_params

    def _gen_random_prop(self, experiment):
        """
        Generates a single random proposal in accordance to experiment.

        Parameters
        ----------
        experiment : experiment
            The experiment representing the current state.

        Returns
        -------
        param_dict_eval : dict
            Dictionary with one string key for each parameter name, and the
            0-1 hypercube value for each of them as value.
        """
        self._logger.log(5, "Generating single random prop for %s", experiment)
        param_defs = experiment.parameter_definitions
        param_dict_eval = {}
        param_names = sorted(param_defs.keys())
        self._logger.log(5, "Generated param_names %s", param_names)
        for pn in param_names:
            pdef = param_defs[pn]
            param_dict_eval[pn] = np.random.uniform(0, 1, pdef.warped_size())
        self._logger.log(5, "Randomly generated %s", param_dict_eval)
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
        self._logger.log(5, "Translating dict %s to vector.", x)
        param_to_eval = []
        param_names = sorted(x.keys())
        for pn in param_names:
            param_to_eval.extend(x[pn])
        self._logger.log(5, "Result is %s", param_to_eval)
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
        self._logger.log(5, "Translating %s from vector to dict. Experiment"
                           " is %s", x_vector, experiment)
        x_dict = {}

        param_names_sorted = sorted(experiment.parameter_definitions.keys())
        warped_lengths = []
        for pn in param_names_sorted:
            warped_lengths.append(experiment.parameter_definitions[pn].warped_size())
        index = 0
        for i, pn in enumerate(param_names_sorted):
            x_dict[pn] = x_vector[index:index+warped_lengths[i]]
            index += warped_lengths[i]
        self._logger.log(5, "Translated to %s", x_dict)
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
        self._logger.log(5, "Translating vector %s to nd_array.", x_vec)
        param_nd_array = np.zeros((1, len(x_vec)))
        for i in range(len(x_vec)):
            param_nd_array[0,i] = x_vec[i]
        self._logger.log(5, "Translated to %s", param_nd_array)
        return param_nd_array

    def in_hypercube(self, x_vec):
        self._logger.log(5, "Testing %s being in hypercube", x_vec)
        for i in range(len(x_vec)):
            if not 0 <= x_vec[i] <= 1:
                self._logger.log(5, "It is not.")
                return False
        self._logger.log(5, "It is.")
        return True


class GradientAcquisitionFunction(AcquisitionFunction):
    """
    This represents an acquisition function whose gradient we can compute.

    This allows us to introduce some more (and nicer) optimizations.
    """


    default_max_searcher = "LBFGSB"
    default_multi_searcher = "random_weighted"


    @abstractmethod
    def gradient(self, x, gp, experiment):
        """
        Computes the gradient of the function at x.

        Signature is the same as evaluate.
        """
        pass

    def _compute_minimizing_gradient(self, x, gp, experiment):
        """
        One problem is that, as a standard, scipy.optimize only searches
        minima. This means we have to convert each acquisition function to
        the minima meaning the best result.
        Whether to actually return the negative gradient or not is set by the
        self.minimizes parameter. If true, it will not change the sign.

        Function signature is as evaluate.
        """
        self._logger.log(5, "Computing minimizing gradient for %s. gp is %s,"
                           "experiment is %s", x, gp, experiment)
        result = self.gradient(x, gp, experiment)
        if self.minimizes:
            self._logger.log(5, "Is minimization. Returning %s", result)
            return result
        else:
            self._logger.log(5, "Is maximizing. Returning %s", -result)
            return -result

    def max_searcher_LBFGSB(self, gp, experiment, good_results=None):
        """
        Searches the maximum proposal via L-BFGS-B.

        For signature see the class docs.
        """
        self._logger.debug("Searching maximum via LBFGSB. gp is %s, "
                           "experiment is %s, good_results %s", gp,
                           experiment, good_results)
        bounds = []
        for pd in experiment.parameter_definitions.values():
            bounds.extend([(0.0, 1.0) for x in range(pd.warped_size())])
        if good_results is None:
            good_results = []
        random_prop = self._gen_random_prop(experiment)
        random_prop_result = self._compute_minimizing_evaluate(random_prop,
                                                               gp, experiment)
        good_results.append((random_prop, random_prop_result))
        self._logger.log(5, "Initialized the first good result. Is %s",
                           good_results)
        scipy_optimizer_results = []

        random_restarts = self.params.get("num_restarts", 10)
        self._logger.debug("Doing %s restarts", random_restarts)
        for i in range(random_restarts):
            self._logger.log(5, "New restart.")
            initial_guess = self._translate_dict_vector(
                self._gen_random_prop(experiment))
            self._logger.log(5, "Initial guess is %s", initial_guess)
            result = scipy.optimize.minimize(
                self._compute_minimizing_evaluate, x0=initial_guess,
                method="L-BFGS-B", jac=self._compute_minimizing_gradient,
                options={'disp': False}, bounds=bounds,
                args=tuple([gp, experiment]))
            self._logger.log(5, "Result of optimization %s", result)
            x_min = result.x
            f_min = result.fun
            success = result.success
            self._logger.log(5, "Success: %s", success)
            if success:
                x_min_dict = self._translate_vector_dict(x_min, experiment)
                if self.in_hypercube(x_min):
                    self._logger.log(5, "Is in hypercube. Appending.")
                    scipy_optimizer_results.append((x_min_dict, f_min))
                else:
                    self._logger.log(5, "Is not in hypercube. Ignoring.")

        scipy_optimizer_results.extend(good_results)
        best_idx = [x[1] for x in scipy_optimizer_results].index(
            min([x[1] for x in scipy_optimizer_results]))
        max_prop = scipy_optimizer_results[best_idx]
        del scipy_optimizer_results[best_idx]
        self._logger.log(5, "Extracted best. Final is max_prop %s, "
                           "scipy_optimizer_results %s", max_prop,
                           scipy_optimizer_results)
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
        self._logger.log(5, "evaluating ExpectedImprovement on %s; gp %s,"
                           " experiment %s", x_vec, gp, experiment)
        x_value = self._translate_vector_nd_array(x_vec)

        #mean, variance and their gradients
        mean, variance = gp.predict(x_value)
        gradient_mean, gradient_variance = gp.predictive_gradients(x_value)
        self._logger.log(5, "Predicted mean/variance of %s / %s. Gradients "
                           "are %s and %s respectively.", mean, variance,
                           gradient_mean, gradient_variance)
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
        self._logger.log(5, "Our best result till now was %s", x_best)
        #handle case of maximization
        sign = 1
        if not experiment.minimization_problem:
            sign = -1

        z_numerator = sign * (x_best - mean + self.params.get(
            "exploitation_exploration_tradeoff", 0))

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
            ei_gradient_part3 = (-1 * gradient_variance * cdf_z * z *
                                 (1/(2*std_dev)))
            ei_gradient = (ei_gradient_part1 + ei_gradient_part2 +
                           ei_gradient_part3)

            ei_gradient = np.transpose(ei_gradient)[0]
        else:
            self._logger.log(5, "std_dev was 0. Returning 0, 0.")
        self._logger.log(5, "ei_value, ei_gradient: %s, %s", ei_value,
                           ei_gradient)
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
        self._logger.log(5, "Evaluating vector to gradient. x_vec %s, gp %s,"
                           " experiment %s", x_vec, gp, experiment)
        value, grad = self._evaluate_vector(x_vec, gp, experiment)
        self._logger.log(5, "Gradient is %s", grad)
        return grad

    def gradient(self, x, gp, experiment):
        self._logger.log(5, "Computing gradient for %s. gp is %s, experiment "
                           "%s", x, gp, experiment)
        if isinstance(x, dict):
            self._logger.log(5, "x is dict. Translating.")
            x_value = self._translate_dict_vector(x)
        else:
            x_value = x
        value, gradient = self._evaluate_vector(x_value, gp, experiment)
        self._logger.log(5, "Evaluated. Returning %s", gradient)
        return gradient

    def evaluate(self, x, gp, experiment):
        self._logger.log(5, "Evaluating %s. gp is %s, experiment %s", x, gp,
                           experiment)
        if isinstance(x, dict):
            x_value = self._translate_dict_vector(x)
            self._logger.log(5, "x was dict, translating to %s", x_value)
        else:
            x_value = x
        value, gradient = self._evaluate_vector(x_value, gp, experiment)
        self._logger.log(5, "Returning %s.", value)
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
        self._logger.log(5, "Evaluating probability of improvement. x is %s,"
                           " gp is %s, experiment %s", x, gp, experiment)
        dimensions = len(experiment.parameter_definitions)
        x_value_vector = self._translate_dict_vector(x)
        x_value = self._translate_vector_nd_array(x_value_vector)

        mean, variance = gp.predict(x_value)
        self._logger.log(5, "Mean and variance are %s, %s", mean, variance)
        # do not standardize on our own, but use the mean, and covariance
        # we get from the gp
        stdv = variance ** 0.5
        x_best = experiment.best_candidate.result
        z = (x_best - mean)/stdv

        cdf = scipy.stats.norm().cdf(z)
        result = cdf
        self._logger.log(5, "Got cdf from scipy.stats. Result is %s", result)
        if not experiment.minimization_problem:
            result = 1 - cdf
            self._logger.log(5, "We're changing because we're maximizing. New "
                               "result is %s", result)
        return result