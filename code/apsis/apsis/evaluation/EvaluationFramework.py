import time
import matplotlib.pylab as plt
import numpy as np
import logging
from apsis.utilities.EvaluationWriter import EvaluationWriter
import random

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
    evaluations: list of dicts
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

                #result tracking
                result_per_step: [result_0, result_1,...],
                best_result_per_step: [best_0, best_1,...],
                cost_eval_per_step: [cost_0, cost_1,...],
                cost_core_per_step: [cost_core0, cost_core1],

                #experiment description
                optimizer: OptimizationCoreInterface,
                objective_function: String
                start_date: timestamp
                end_date: timestamp

                #internal attributes, used and set by EvaluationWriter
                _output_folder: string with directory absolute path
                _steps_written: int
            }

    plots: list of dicts
        A list containing dicts to store the plots created in memery. They
        will be store here until they are written by EvaluationWriter which
        will after writing remove them from this list.

        {
            'step': int - the step at which this plot was created
            'plots': {
                'plot_name': plt.figure obj,
                'other_plot_name': plt.figure obj,
            }
        }
    """
    evaluations = None

    plots = None

    COLORS = ["g", "r", "c", "b", "m", "y"]
    evaluation_writer = None

    def __init__(self):
        """
        Constructor of EvaluationFramework. Only instantiation evaluations
        and plot storage list here.
        """
        self.evaluations = []
        self.plots = []

        #create csv writer
        eval_framework = self
        self.evaluation_writer = EvaluationWriter(eval_framework)

    def evaluate_and_plot_precomputed_grid(self, optimizers,
                                           evaluation_descriptions, grid,
                                           steps, to_plot=None):
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
        self.evaluate_precomputed_grid(optimizers, evaluation_descriptions,
                                       grid, steps)
        self.plot_evaluations(to_plot=to_plot, store=False,
                              show_after_creation=True)

    def evaluate_precomputed_grid(self, optimizers, evaluation_descriptions,
                                  grid, steps):
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
        self.evaluate_optimizers(optimizers, evaluation_descriptions,
                                 grid.evaluate_candidate,
                                 obj_func_name="precomputed_grid", steps=steps)



    def evaluate_optimizers(self, optimizers, evaluation_descriptions,
                            objective_function, obj_func_name=None, steps=20,
                            write_csv=True,  write_detailed_results=True,
                            csv_write_frequency=10, plot_write_frequency=10):
        """
        Unconstrained evaluation of all optimizers on the given objective func.
        In each step evaluates all optimizers given for exactly one step.

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

        write_csv: boolean
            If this evaluation run shall be appended to the global reporting,
            CSV. See utilities.EvaluationWriter for details.

        write_detailed_results: boolean
            If this evaluation run shall report all results including plots to
            the output folder declared in EvaluationWriter.

        csv_write_frequency: int or None
            How often the detailed output to csv will be written during
            evaluation. Argument gives number of steps.
            To make it write only at the end assign None here.

            Default: 10

        plot_write_frequency: int or None
            How often the plot output will be written during
            evaluation. Argument gives number of steps.
            To make it write only at the end assign None here.

            Default: 10
        """
        #create a new evaluation hash for every optimizer.
        optimizer_idxs = self._add_new_optimizer_evaluation(optimizers,
                                                    evaluation_descriptions,
                                                    obj_func_name)
        #then in each step optimize with each optimizer just for one step.
        for i in range(steps):
            for optimizer_idx in optimizer_idxs:
                self.evaluation_step(optimizer_idx, objective_function)

            #write out the detailed reseults in a certain frequency if whished
            if write_detailed_results:
                if csv_write_frequency is not None and i % csv_write_frequency == 0:
                    self.evaluation_writer.append_evaluations_to_detailed_csv()

                if plot_write_frequency is not None and i % plot_write_frequency == 0:
                    self.evaluation_writer.append_evaluations_to_detailed_csv()

        #insert end date to all experiments
        for ev in self.evaluations:
            ev['end_date'] = time.time()


        #automatically save this run to global csv
        if write_csv:
            try:
                self.evaluation_writer.write_evaluations_to_global_csv()
                logging.info("Wrote summary CSV output - Finished.")
            except ValueError:
                logging.error("Error writing result to global CSV file after "
                              "finishing evaluations.")

        if write_detailed_results:
            logging.info("Wrote detailed CSV output - Finished.")
            #write detailed csv
            self.evaluation_writer.append_evaluations_to_detailed_csv()
            #write plots
            self.evaluation_writer.write_out_plots_all_evaluations()

            logging.info("Wrote detailed CSV output and plots - Finished.")



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
        #retrieve the optimizer according to its index
        optimizer = self.evaluations[core_index]['optimizer']

        #let the optimizer offer a new candidate to us
        #compute next candidate - track cost
        start_time = time.time()
        next_candidate = optimizer.next_candidate()
        cost_core = time.time() - start_time

        #evaluate this candidate in objective function.
        # It has to update result and other properties in next_candidate.
        next_candidate = objective_func(next_candidate)

        #Now we report back the evaluation result.
        #also the cost for the working method is accounted to the optimizer
        start_time = time.time()
        optimizer.working(next_candidate, "finished")
        cost_core += time.time() - start_time

        best_result = optimizer.best_candidate.result

        #storing this evaluation step.
        self._add_evaluation_step(core_index, next_candidate.result,
                                  best_result, next_candidate.cost, cost_core)

    def plot_evaluation_step_ranking(self, idxs=None, show_after_creation=False):
        if idxs is None:
            idxs = range(len(self.evaluations))

        x_list = []
        y_list = []
        y_format = []
        x_label = "Number of Evaluations of Objective Function"
        y_label = "Ranking of result"
        for i, idx in enumerate(idxs):
            color = self.COLORS[i%len(self.COLORS)]#float(i)/len(idxs)
            desc = self.evaluations[idx]['description']
            y_format.append({
                "type": "scatter",
                "label": desc,
                "color": color
            })
            result_list_sort = self.evaluations[idx]['result_per_step'][:]
            result = []
            result_list_sort.sort()
            for r in self.evaluations[idx]['result_per_step']:
                result.append(result_list_sort.index(r))
            y_list.append(result)
            num_steps = len(result)

            x_list.append(np.linspace(0, num_steps, num_steps, endpoint=False))
        return self._plot_lists(x_list, y_list, y_format, x_label, y_label,
                                show_after_creation=show_after_creation)


    def plot_evaluations_best_result_by_num_steps(self, idxs=None,
                                                  show_after_creation=False):
        """
        Create a 2D plot with the following setting
            X-Axis: Number of evaluations of objective functions (= num steps)
            Y-Axis: Line: Best objective function value achieved so far.
                    Dots: Result for that evaluation

        Parameters
        ----------

        idxs: list of int
            A list of indexes corresponding to the indexes of evaluations
            for which evaluations a plot shall be created.
        """
        if idxs is None:
            idxs = range(len(self.evaluations))

        x_list = []
        y_list = []
        y_format = []
        x_label = "Number of Evaluations of Objective Function"
        y_label = "Best Objective Function Result"
        for i, idx in enumerate(idxs):
            color = self.COLORS[i%len(self.COLORS)]#float(i)/len(idxs)
            desc = self.evaluations[idx]['description']
            y_format.append({
                "type": "line",
                "label": desc,
                "color": color
            })
            results = self.evaluations[idx]['best_result_per_step']
            y_list.append(results)
            num_steps = len(results)

            x_list.append(np.linspace(0, num_steps, num_steps, endpoint=False))

            desc = self.evaluations[idx]['description']
            y_format.append({
                "type": "scatter",
                "label": desc,
                "color": color
            })
            results = self.evaluations[idx]['result_per_step']
            y_list.append(results)
            num_steps = len(results)

            x_list.append(np.linspace(0, num_steps, num_steps, endpoint=False))
        return self._plot_lists(x_list, y_list, y_format,
                                x_label, y_label,
                                show_after_creation=show_after_creation)


    def _plot_lists(self, x_list, y_list, y_format=None, x_label=None,
                    y_label=None, show_after_creation=False):
        this_plot = plt.figure()
        if y_format is None:
            y_format = []
            for i in range(len(y_list)):
                y_format.append({})

        if x_label is not None:
            plt.xlabel(x_label)
        if y_label is not None:
            plt.ylabel(y_label)

        for i, y in enumerate(y_list):
            type = y_format[i].get("type", "line")
            label = y_format[i].get("label", "")
            color = y_format[i].get("color", random.random())
            if type == "line":
                plt.plot(x_list[i], y, label=label, color=color)
            elif type == "scatter":
                plt.scatter(x_list[i], y, label=label, color=color)
        plt.legend(loc='upper right')

        if(show_after_creation):
            plt.show(True)

        return this_plot




    def plot_evaluations_best_result_by_cost(self, idxs=None,
                                             show_after_creation=False):
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

        Returns
        -----------

        matplotlib.figure
            The figure object representing this plot.
        """
        if idxs is None:
            idxs = range(len(self.evaluations))

        x_list = []
        y_list = []
        y_format = []
        x_label = "Cost of Total Optimization"
        y_label = "Best Objective Function Result"
        for i, idx in enumerate(idxs):
            #compute the total cost as total_cost = eval_cost + core_cost
            total_costs = []
            cost_before = 0
            for step in range(len(self.evaluations[idx]['best_result_per_step'])):
                total_costs.append(self.evaluations[idx]['cost_eval_per_step']
                                   [step] + self.evaluations[idx]
                                   ['cost_core_per_step'][step]
                                   + cost_before)
                cost_before = total_costs[step]

            color = self.COLORS[i%len(self.COLORS)]#float(i)/len(idxs)
            desc = self.evaluations[idx]['description']
            y_format.append({
                "type": "line",
                "label": desc,
                "color": color
            })
            results = self.evaluations[idx]['best_result_per_step']
            y_list.append(results)

            x_list.append(total_costs)

            desc = self.evaluations[idx]['description']
            y_format.append({
                "type": "scatter",
                "label": desc,
                "color": color
            })
            results = self.evaluations[idx]['result_per_step']
            y_list.append(results)
            x_list.append(total_costs)
        return self._plot_lists(x_list, y_list, y_format, x_label, y_label,
                                show_after_creation=show_after_creation)

    def plot_evaluations(self, idxs=None, to_plot=None,
                         show_after_creation=False, store=True):
        """
        Creates the plots defined in to_plot.

        Parameters
        ----------

        idxs: list of int
            A list of indexes corresponding to the indexes of evaluations
            for which evaluations a plot shall be created.

        to_plot: list of string or string
            Possible plot options. Plots all plots corresponding to the strings
             in the list. They can be in any order, and invalid values will be
             ignored.

            Possible values:
                best_result_per_step
                best_result_per_cost

            Default option is all.
        """
        plot_all = False
        if to_plot is None:
            plot_all = True
        elif not isinstance(to_plot, list):
            to_plot = [to_plot]

        plots_to_store = {}

        #if we have a new plot we need to add it here
        if "best_result_per_step" in to_plot or plot_all:
            best_result_per_step_plt = \
                self.plot_evaluations_best_result_by_num_steps(idxs,
                                    show_after_creation=show_after_creation)

            #either store plot or close to free memory
            if store:
                plots_to_store['best_result_per_step'] = best_result_per_step_plt
            else:
                plt.close(best_result_per_step_plt)

        if "best_result_per_cost" in to_plot or plot_all:
            best_result_per_cost_plt = \
                self.plot_evaluations_best_result_by_cost(idxs,
                                    show_after_creation=show_after_creation)
            if store:
                plots_to_store['best_result_per_cost'] = best_result_per_cost_plt
            else:
                plt.close(best_result_per_cost_plt)

        if "plot_evaluation_step_ranking" in to_plot or plot_all:
            plot_evaluation_step_ranking_plt = \
                self.plot_evaluation_step_ranking(idxs,
                                    show_after_creation=show_after_creation)

            if store:
                plots_to_store["plot_evaluation_step_ranking"] = \
                    plot_evaluation_step_ranking_plt
            else:
                plt.close(plot_evaluation_step_ranking_plt)

        if store:
            self._store_plots(plots_to_store)


    def _store_plots(self, plot_dict):
        """
        Convenience methods to store all plots in the current step. Using this
        method only makes sense if you use the EvaluationFramework to either
        evaluate only one OptimizationCore, or evaluate several cores
        simultaneously using the EvaluationFramework.evaluate_optimizers()
        function.

        Warning: The step that is stored with the plots is calculated naively as the
        current step of an arbitrary optimizer evaluation has in the
        evaluations list.

        Parameters
        ----------
            plot_dict: dict
                A dict containing the plot name as keys and the
                matplotlib.figure object as values.

        """
        step_to_store = len(self.evaluations[0]['result_per_step'])

        self.plots.append({
            'step': step_to_store,
            'plots': plot_dict
        })

    def _add_evaluation_step(self, core_index, result, best_result, cost_eval,
                             cost_core):
        dict_to_update = self.evaluations[core_index]

        dict_to_update['result_per_step'].append(result)
        dict_to_update['best_result_per_step'].append(best_result)
        dict_to_update['cost_eval_per_step'].append(cost_eval)
        dict_to_update['cost_core_per_step'].append(cost_core)

    def _add_new_optimizer_evaluation(self, optimizers,
                                      evaluation_descriptions, objective_func_name):
        optimizer_idxs = []
        for i in range(len(optimizers)):
            optimizer_dict = {
                'description': evaluation_descriptions[i],
                'optimizer': optimizers[i],
                'result_per_step': [],
                'best_result_per_step': [],
                'cost_eval_per_step': [],
                'cost_core_per_step': [],
                'objective_function': objective_func_name,
                'start_date': time.time(),
                'end_date':None
            }

            optimizer_idxs.append(len(self.evaluations))
            self.evaluations.append(optimizer_dict)

        return optimizer_idxs














