__author__ = 'Frederik Diehl'

from apsis.utilities.benchmark_functions import *
from pylab import *
from apsis.assistants.lab_assistant import ValidationLabAssistant
from apsis.models.parameter_definition import *

def learn_one_var(opt, LAss, dims, points, var, steps, noise_gen, exp_name):
    """
    This learns one variance for the optimizer.

    Parameters
    ----------
    opt : Optimizer
        The optimizer to evaluate.
    LAss : ValidationLabAssistant
        The LAss used for this evaluation. This will be updated.
    dims : int
        The dimension count over which to evaluate.
    points : int
        The number of points per dimension for the noise.
    var : float
        The variance on which to evaluate
    steps : int
        The number of steps for each of the optimizers to learn. Note that each
        single experiment will only be updated steps/cv times.
    noise_gen : np.ndarray
        The noise generation array.
    exp_name : string
        Name for the experiment for further reference.
    """
    val_min, val_max = 0, 1
    param_def = {}
    for d in range(dims):
        param_def[str(d)] = MinMaxNumericParamDef(0, 1)
    LAss.init_experiment(exp_name, opt, param_def)
    for s in range(steps):
        cand = LAss.get_next_candidate(exp_name)
        params_list = []
        for d in range(dims):
            params_list.append(cand.params[str(d)])
        cand.result = get_noise_value_at(params_list, variance=var, noise_gen=noise_gen,
                                       val_min=val_min, val_max=val_max)
        LAss.update(exp_name, cand)

def evaluate_one_opt(opt, LAss, dims, points, min_var, max_var, step_var, steps, noise_gen):
    """
    Evaluates one optimizer on several variances.

    Parameters
    ----------
    opt : Optimizer
        The optimizer to evaluate.
    LAss : ValidationLabAssistant
        The LAss used for this evaluation. This will be updated.
    dims : int
        The dimension count over which to evaluate.
    points : int
        The number of points per dimension for the noise.
    min_var : float
        The minimum variance to evaluate.
    max_var : float
        The maximum variance to evaluate.
    step_var : float
        The distance between two variance values to evaluate.
    steps : int
        The number of steps for each of the optimizers to learn. Note that each
        single experiment will only be updated steps/cv times.

    Returns
    -------
    performances : list of floats
        The mean performances for each variance
    variance : list of floats
        The performance variance for each of the noise variances. Same length
        as performances.
    """
    performances = []
    variances = []
    for i, var in enumerate(np.arange(min_var, max_var, step_var)):
        exp_name = str(opt) + "_" + str(var)
        learn_one_var(opt, LAss, dims, points, var, steps, noise_gen, exp_name)
        best_cands = LAss.get_best_candidates(exp_name)
        performances.append(np.mean([x.result for x in best_cands]))
        variances.append(np.var([x.result for x in best_cands]))
    return performances, variances

def evaluate_performance(optimizers, dims, points, min_var, max_var, step_var, steps, cv):
    """
    Evaluates the performance of the optimizers over several variances.

    Parameters
    ----------
    optimizers : list of Optimizers
        The optimizers to evaluate.
    dims : int
        The dimension count over which to evaluate.
    points : int
        The number of points per dimension for the noise.
    min_var : float
        The minimum variance to evaluate.
    max_var : float
        The maximum variance to evaluate.
    step_var : float
        The distance between two variance values to evaluate.
    steps : int
        The number of steps for each of the optimizers to learn.
    cv : int
        The crossvalidation number for comparison purposes.
    """
    LAss = ValidationLabAssistant(cv=cv)
    performances = {}
    variances = {}
    noise_gen = gen_noise(dims, points, random_state=42)
    for o in optimizers:
        print(o)
        performances[o], variances[o] = evaluate_one_opt(o, LAss, dims, points, min_var, max_var, step_var, steps*cv, noise_gen)
    var_space = np.arange(min_var, max_var, step_var)
    plt.xlabel("smoothing gaussian variance")
    plt.ylabel("best result after %i steps" %steps)
    plt.title("Performance of optimization in dependance on the smoothness.")
    plt.ylim((0, 1))
    for o in optimizers:

        plt.errorbar(var_space, performances[o], label=str(o), yerr=variances[o], linewidth=2.0, capthick=4, capsize=8.0)#, fmt='o'
    plt.legend(loc='lower right')
    plt.show(True)


def learn_on_noise(optimizers, dims, points, var, steps, cv, show_plot=True):
    """
    Evaluates the optimizers on one specific variance.

    Parameters
    ----------
    optimizers : list of optimizers
        The optimizers to compare.
    dims : int
        The number of dimensions on which to run.
    points : int
        The number of points per dimension for the noise generation.
    var : float
        The variance for the gaussian with which the noise is smoothed.
    steps : int
        The number of steps for the optimizers to run
    cv : int
        The crossvalidation number for comparison purposes.
    show_plot : bool, optional
        Whether to show the plot in the end. Default is True.
    """
    LAss = ValidationLabAssistant(cv=cv)
    exp_names = []
    noise_gen = gen_noise(dims, points, random_state=42)
    for opt in optimizers:
        exp_names.append(str(opt) + "_" + str(var))
        learn_one_var(opt, LAss, dims, points, var, steps*cv, noise_gen, exp_names[-1])
    for e in exp_names:
        print(str(e) + ": " + str(sort([x.result for x in LAss.get_best_candidates(e)])))

    if show_plot:
        LAss.plot_validation(exp_names, plot_min=0, plot_max=1)


if __name__ == '__main__':
    optimizers = ["RandomSearch", "BayOpt"]
    dims = 3
    points = 50
    evaluate_performance(optimizers, dims, points, min_var=0.001,
                                 max_var=0.016, step_var=0.001, steps=20, cv=10)