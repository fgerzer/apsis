import time
import matplotlib.pylab as plt
from matplotlib.lines import Line2D
import numpy as np

class EvaluationFramework(object):
    """
    Documentation TBD

    Attributes
    ----------
    evaluations
        Store list of evaluation dicts. An evaluation dict contains
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
        self.evaluations = []

    def plot_precomputed_grid(self, optimizers, evaluation_descriptions, grid, steps):
        self.evaluate_optimizers_precomputed_grid(optimizers, evaluation_descriptions, grid, steps)
        self.plot_evaluations()

    def evaluate_optimizers_precomputed_grid(self, optimizers, evaluation_descriptions, grid, steps):
        self.evaluate_optimizers(optimizers, evaluation_descriptions, grid.evaluate_candidate, steps)

    def evaluate_optimizers(self, optimizers, evaluation_descriptions,
                            objective_function, steps):

        optimizer_idxs = self._add_new_optimizer_evaluation(optimizers,
                                                    evaluation_descriptions)

        for i in range(steps):
            for optimizer_idx in optimizer_idxs:
                self.evaluation_step(optimizer_idx, objective_function)


    def evaluation_step(self, core_index, objective_func):
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

        best_result = optimizer.best_result

        self._add_evaluation_step(core_index, next_candidate.result,
                                  best_result, next_candidate.cost, cost_core)


    def plot_evaluations(self, idxs=None):
        plt.hold()

        if idxs is None:
            idxs = range(len(self.evaluations))

        for idx in idxs:
            results = self.evaluations[idx]['result_per_step']
            desc = self.evaluations[idx]['description']
            num_steps = len(results)
            x = np.linspace(0, num_steps, num_steps, endpoint=False)

            plt.plot(x, results, label=desc)

        plt.legend(loc='upper right')
        plt.show()


    def _add_evaluation_step(self, core_index, result, best_result, cost_eval, cost_core):
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


