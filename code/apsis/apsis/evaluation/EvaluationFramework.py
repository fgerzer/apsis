import time
import matplotlib.pylab as plt
from matplotlib.lines import Line2D
import numpy as np
import logging

class EvaluationFramework(object):
    """
    Evaluation Framework for evaluation one or a number of optimizers.

    Needs a list of instantiated optimizers as a basis and evaluates a defined
    number of points with these optimizers. In each step it continues
    each optimizer for one step in order to be able to investigate the progress
    of all optimizers simultaneously.

    It keeps track of all performed evaluations in its list of evaluation
    hashes (see Attributes section).

    Attributes
    ----------
    evaluations
        Store list of evaluation dicts. An evaluation dict is a dictionary that
        contains information about the performance evaluation of a particular
        instantiated optimizer. It maintains lists of achieved results on the
        objective function as well as the series of best_results and costs
        occurred on optimization. Cost is being tracked in milliseconds and is
        made up by the costs occurring on evaluation of objective and by
        the summed costs for execution of the two methods next_candidate and
        working in the optimizer.
        The description string is used to describe each evaluation run within
        the plots and printouts

        Evaluation dict looks as follows
            {
                description: String
                optimizer: OptimizationCoreInterface,
                result_per_step: [result_0, result_1,...],
                best_result_per_step: [best_0, best_1,...],
                cost_eval_per_step: [cost_0, cost_1,...],
                cost_core_per_step: [cost_core0, cost_core1]
            }
    """
    evaluations = None

    def __init__(self):
        """
        Constructor of EvaluationFramework. Only instantiation evaluations
        list here.
        """
        self.evaluations = []

    def evaluate_and_plot_precomputed_grid(self, optimizers, evaluation_descriptions, grid, steps):
        """
        Evaluates all optimizers on the given precomputed grid and plots
        the results for all of them.

        Parameters
        ----------
        optimizers: OptimizationCoreInterface
            The optimizers used for evaluation. The optimizers need to be
            instantiated before.
        evaluation_descriptions: list of strings
            Contains a string fro each evaluation of an optimizer to describe
            this evaluation e.g. "SimpleBayesian_EI_scipy_brute". Will be
            used in plotting
        grid: PreComputedGrid
            Method to evaluate and plot all optimizers given on the grid given
            in grid.
            Hence you have to instantiate the grid and run build_grid_points
            and precompute_results. Exmaple:
               grid.build_grid_points(param_defs=param_defs, dimensionality=10)
               grid.precompute_results(func)
        steps: int
            The number of steps to run the evaluation for.
        """
        self.evaluate_precomputed_grid(optimizers, evaluation_descriptions, grid, steps)
        self.plot_evaluations()

    def evaluate_precomputed_grid(self, optimizers, evaluation_descriptions, grid, steps):
        """
        Evaluates all optimizers on the given precomputed grid. You will be
        able to investigate results in the local evaluations attribute.

        Parameters
        ----------
        optimizers: OptimizationCoreInterface
            The optimizers used for evaluation. The optimizers need to be
            instantiated before.
        evaluation_descriptions: list of strings
            Contains a string fro each evaluation of an optimizer to describe
            this evaluation e.g. "SimpleBayesian_EI_scipy_brute". Will be
            used in plotting
        grid: PreComputedGrid
            Method to evaluate and plot all optimizers given on the grid given
            in grid.
            Hence you have to instantiate the grid and run build_grid_points
            and precompute_results. Exmaple:
               grid.build_grid_points(param_defs=param_defs, dimensionality=10)
               grid.precompute_results(func)
        steps: int
            The number of steps to run the evaluation for.
        """
        self.evaluate_optimizers(optimizers, evaluation_descriptions, grid.evaluate_candidate, steps)



    def evaluate_optimizers(self, optimizers, evaluation_descriptions,
                            objective_function, steps):
        """
        Unconstrained evaluation of all optimizers on the given objective func.

        Please read carefully about the semantics of the objective function
        below.

        You will be able to investigate results in the local
        evaluations attribute.

        Parameters
        ----------
        optimizers: OptimizationCoreInterface
            The optimizers used for evaluation. The optimizers need to be
            instantiated before.
        evaluation_descriptions: list of strings
            Contains a string fro each evaluation of an optimizer to describe
            this evaluation e.g. "SimpleBayesian_EI_scipy_brute". Will be
            used in plotting
        objective_function: function
            A function with the following signature
                objective(candidate): Candidate
            taking a Candidate object as only argument and returning a
            candidate object with the result value set.
        steps: int
            The number of steps to run the evaluation for.
        """
        optimizer_idxs = self._add_new_optimizer_evaluation(optimizers,
                                                    evaluation_descriptions)

        for i in range(steps):
            for optimizer_idx in optimizer_idxs:
                self.evaluation_step(optimizer_idx, objective_function)


    def evaluation_step(self, core_index, objective_func):
        """
        Performs one evaluation step with the optimizer identified by the
        index of the specific evaluation run in the list of evaluations held
        by this class.

        Parameters
        ----------

        core_index: int
            The index of the optimizer according to its index in the
            list of evaluations held in evaluations.

        objective_function: function
            A function with the following signature
                objective(candidate): Candidate
            taking a Candidate object as only argument and returning a
            candidate object with the result value set.

        """
        optimizer = self.evaluations[core_index]['optimizer']

        #compute next candidate - track cost
        start_time = time.time()
        next_candidate = optimizer.next_candidate()
        cost_core = time.time() - start_time

        #evaluate in objective function. It has to update result and other
        #properties in next_candidate.
        next_candidate = objective_func(next_candidate)
        #also the cost for the working method is accounted to the optimizer
        start_time = time.time()
        optimizer.working(next_candidate, "finished")
        cost_core += time.time() - start_time

        best_result = optimizer.best_candidate.result

        self._add_evaluation_step(core_index, next_candidate.result,
                                  best_result, next_candidate.cost, cost_core)


    def plot_evaluations_best_result_by_num_steps(self, idxs=None):
        """
        Create a 2D plot with the following setting
            X-Axis: Number of evaluations of objective functions (= num steps)
            Y-Axis: Best objective function value achieved so far.

        Parameters
        ----------

        idxs: list of int
            A list of indexes corresponding to the indexes of evaluations
            for which evaluations a plot shall be created.
        """
        if idxs is None:
            idxs = range(len(self.evaluations))

        plt.figure()
        for idx in idxs:
            results = self.evaluations[idx]['best_result_per_step']
            desc = self.evaluations[idx]['description']
            num_steps = len(results)
            x = np.linspace(0, num_steps, num_steps, endpoint=False)
            logging.debug("Plotting %s (optimizer %s), with results %s"
                          %(desc, str(self.evaluations[idx]["optimizer"]), str(results)))
            plt.plot(x, results, label=desc)

        plt.legend(loc='upper right')
        plt.show(True)

    def plot_evaluations_best_result_by_cost(self, idxs=None):
        """
        Create a 2D plot with the following setting
            X-Axis: Total cost of parameter optimization
                total cost = cost of evaluation + cost of optimization logic.
            Y-Axis: Best objective function value achieved so far.

        Parameters
        ----------

        idxs: list of int
            A list of indexes corresponding to the indexes of evaluations
            for which evaluations a plot shall be created.
        """
        if idxs is None:
            idxs = range(len(self.evaluations))

        #plot costs
        plt.figure()
        for idx in idxs:
            total_costs = []
            cost_before = 0
            for step in range(len(self.evaluations[idx]['best_result_per_step'])):
                total_costs.append(self.evaluations[idx]['cost_eval_per_step'][step]+ self.evaluations[idx]['cost_core_per_step'][step] + cost_before)
                cost_before = total_costs[step]

            results = self.evaluations[idx]['best_result_per_step']
            desc = self.evaluations[idx]['description']



            x = total_costs
            logging.debug("Plotting %s (optimizer %s), results %s, with cost %s"
                          %(desc, str(self.evaluations[idx]["optimizer"]), str(results), str(total_costs)))
            plt.plot(x, results, label=desc)

        plt.show(True)

    def plot_evaluations(self, idxs=None):
        """
        Creates the plots that is given by the following two functions in this
        order.
            1. plot_evaluations_best_result_by_num_steps
            2. plot_evaluations_best_result_by_cost

        Parameters
        ----------

        idxs: list of int
            A list of indexes corresponding to the indexes of evaluations
            for which evaluations a plot shall be created.

        """
        self.plot_evaluations_best_result_by_num_steps(idxs)
        self.plot_evaluations_best_result_by_cost(idxs)


    def _add_evaluation_step(self, core_index, result, best_result, cost_eval,
                             cost_core):
        dict_to_update = self.evaluations[core_index]

        dict_to_update['result_per_step'].append(result)
        dict_to_update['best_result_per_step'].append(best_result)
        dict_to_update['cost_eval_per_step'].append(cost_eval)
        dict_to_update['cost_core_per_step'].append(cost_core)

    def _add_new_optimizer_evaluation(self, optimizers, evaluation_descriptions):
        optimizer_idxs = []
        for i in range(len(optimizers)):
            optimizer_dict = {
                'description': evaluation_descriptions[i],
                'optimizer': optimizers[i],
                'result_per_step': [],
                'best_result_per_step': [],
                'cost_eval_per_step': [],
                'cost_core_per_step': []
            }

            optimizer_idxs.append(len(self.evaluations))
            self.evaluations.append(optimizer_dict)

        return optimizer_idxs


