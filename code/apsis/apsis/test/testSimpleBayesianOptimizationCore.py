from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from apsis.adapters.SimpleScikitLearnAdapter import SimpleScikitLearnAdapter
from sklearn.metrics import mean_squared_error, mean_absolute_error
from apsis.models.ParamInformation import NumericParamDef, NominalParamDef, \
    LowerUpperNumericParamDef
from apsis.SimpleBayesianOptimizationCore import SimpleBayesianOptimizationCore
import math
import nose.tools as nt
import random
import logging
import matplotlib.pyplot as plt
import numpy as np
import GPy

class testSimpleBayesianOptimizationCore(object):

    def test_setup(self):
        logging.basicConfig(level=logging.DEBUG)
        boston_data = datasets.load_boston()
        regressor = LogisticRegression()

        param_defs = {
            "C": LowerUpperNumericParamDef(0.00001, 10)
        }

        sk_adapter = SimpleScikitLearnAdapter(regressor, param_defs,
                                              scoring="mean_squared_error",
                                              n_iter=20,
                                              cv=5,
                                              optimizer='SimpleBayesianOptimizationCore',
                                              optimizer_arguments={
                                                  'initial_random_runs': 5,
                                                  'num_gp_restarts': 10,
                                                  'minimization': False}
                                              )

        fitted = sk_adapter.fit(boston_data.data, boston_data.target)
        print(fitted.get_params())
        print("Final MSE: " + str(mean_squared_error(boston_data.target,
                                 fitted.predict(boston_data.data))))

        print("Final MAE: " + str(mean_absolute_error(boston_data.target,
                                 fitted.predict(boston_data.data))))

        for c in sk_adapter.optimizer.finished_candidates:
            print("- " + str(c.params) + ": " + str(c.result))

    def test_convergence_one_worker(self):
        min_val = 0
        max_val = 10
        resolution = 1000


        logging.basicConfig(level=logging.DEBUG)
        self.bay_search = SimpleBayesianOptimizationCore({"param_defs":
            [LowerUpperNumericParamDef(min_val, max_val)],
            "initial_random_runs": 5, 'num_gp_restarts': 10})
        strings = []
        f = function
        best_result = None

        for i in range(20):
            cand = self.bay_search.next_candidate()

            point = cand.params
            value = f(point[0])
            if (i >= 5):

                self.plot_nicely(min_val, max_val, resolution, point[0])
                print(self.bay_search.gp)
                raw_input()

            strings.append(("%i: %f at %f" % (i, value, point[0])))
            if best_result is None or value < best_result:
                best_result = value

            cand.result = value
            assert not self.bay_search.working(cand, "finished")

        nt.eq_(self.bay_search.best_candidate.result, best_result,
                       str(self.bay_search.best_candidate.result)
                       + " != " + str(best_result))
        #self.bay_search.gp.plot()
        self.plot_nicely(min_val, max_val, resolution)
        for s in strings:
            print(s)
        raw_input()


    def plot_nicely(self, min_val, max_val, resolution, next_pt=None):
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
            mean, variance, _025pm, _975pm = self.bay_search.gp.predict(cur_format)
            gp_mean.append(mean)
            gp_975.append(_975pm)
            gp_025.append(_025pm)
            acquisition_params = {
                'param_defs': self.bay_search.param_defs,
                'gp': self.bay_search.gp,
                'cur_max': self.bay_search.best_candidate.result
                }
            eval = self.bay_search.acquisition_function.evaluate(warped_cur, acquisition_params)
            acq.append(eval)
            func.append(function(cur))
            cur += step_res
            warped_cur += 1./resolution

        acq_scale = 1
        max_acq = max([X[0, 0] for X in acq])
        max_gp = float(max([X[0, 0] for X in gp_mean]))
        max_func = float(max(func))
        acq_scale = max_acq / max(max_gp, max_func)
        do_scale = True
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
        for c in self.bay_search.finished_candidates:
            pts.append(c.params[0])
            vals.append(c.result)

        plt.scatter(pts, vals)

        if (next_pt is not None):
            plt.scatter([next_pt], [function(next_pt)], color="red")
            print("Next pt: %s, for %f acq value" %(str(next_pt), self.bay_search.acquisition_function.evaluate((next_pt-min_val)/(max_val-min_val), acquisition_params)))
            print("Comp pt: %s, for %f acq value" %(str(0.01), self.bay_search.acquisition_function.evaluate((0.01-min_val)/(max_val-min_val), acquisition_params)))
        else:
            print("Next pt is None.")

        plt.plot(axis, func, color="yellow", label="objective")
        plt.legend(loc='upper left')
        plt.show()

def function(x):
    return math.sin(x) * x**2