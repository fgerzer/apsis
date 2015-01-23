__author__ = 'Frederik Diehl'

import matplotlib.pyplot as plt
from apsis.optimizers.bayesian_optimization import SimpleBayesianOptimizer
from apsis.assistants.experiment_assistant import PrettyExperimentAssistant
from apsis.models.parameter_definition import *
import numpy as np

def function(x):
    return x**3 - 5 * x**2#math.sin(x) * x**2

def test_convergence_one_worker():
    min_val = 0
    max_val = 1
    param_defs = {
        "x": MinMaxNumericParamDef(min_val, max_val)
    }

    BAss = PrettyExperimentAssistant("test", "BayOpt", param_defs=param_defs)
    results = []

    for i in range(50):
        to_eval = BAss.get_next_candidate()
        print(to_eval)
        result = function(to_eval.params["x"])
        results.append(result)
        to_eval.result = result
        BAss.update(to_eval)
        gp = BAss.optimizer.gp
        acq = BAss.optimizer.acquisition_function
        if (i >= 10):
            #plot_nicely(gp, acq, BAss.experiment, min_val, max_val, 100, to_eval.params["x"], True)
            BAss.optimizer.gp.plot()
            print(BAss.optimizer.gp)
            raw_input()


def plot_nicely(gp, acqf, experiment, min_val, max_val, resolution, next_pt=None, minimization=True):
        step_res = (max_val - min_val)/float(resolution)
        gp_mean = []
        gp_975 = []
        gp_025 = []
        acq = []
        func = []
        axis = []
        plt.figure()

        cur = float(min_val)
        warped_cur = 0
        for s in range(resolution):
            axis.append(cur)

            cur_format = np.zeros((1, 1))
            cur_format[0, 0] = warped_cur
            mean, variance = gp.predict(cur_format)
            gp_mean.append(mean)
            #gp_975.append(_975pm)
            gp_975.append(mean + variance)
            gp_025.append(mean - variance)
            #gp_025.append(_025pm)

            eval = acqf.evaluate({"x": s}, gp, experiment=experiment)
            acq.append(eval)
            func.append(function(cur))
            cur += step_res
            warped_cur += 1./resolution

        acq_scale = 1
        max_acq = max([X[0, 0] for X in acq])
        max_gp = float(max([X[0, 0] for X in gp_mean]))
        max_func = float(max(func))
        acq_scale = max_acq / max(max_gp, max_func)
        do_scale = False
        print(max(acq))

        if do_scale:
            acq = [X[0, 0]/acq_scale for X in acq]
        else:
            acq = [X[0, 0] for X in acq]
        plt.plot(axis, [X[0, 0] for X in gp_mean], color="blue", label="mean")
        plt.plot(axis, [X[0, 0] for X in gp_975], color="green", label="gp_975")
        plt.plot(axis, [X[0, 0] for X in gp_025], color="green", label="gp_025")
        plt.plot(axis, acq, color="red", label="acq")
        pts = []
        vals = []
        for c in experiment.candidates_finished:
            pts.append(c.params["x"])
            vals.append(c.result)

        plt.scatter(pts, vals)

        if (next_pt is not None):
            plt.scatter([next_pt], [function(next_pt)], color="red")
            #print("Next pt: %s, for %f acq value" %(str(next_pt), self.bay_search.acquisition_function.evaluate((next_pt-min_val)/(max_val-min_val), acquisition_params)))
            #print("Comp pt: %s, for %f acq value" %(str(0.01), self.bay_search.acquisition_function.evaluate((0.01-min_val)/(max_val-min_val), acquisition_params)))
        else:
            print("Next pt is None.")

        plt.plot(axis, func, color="yellow", label="objective")
        plt.legend(loc='upper left')
        plt.show()

if __name__ == '__main__':
    test_convergence_one_worker()