from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.svm import NuSVC, SVC

from apsis.adapters.SimpleScikitLearnAdapter import SimpleScikitLearnAdapter
from apsis.SimpleBayesianOptimizationCore import SimpleBayesianOptimizationCore
from apsis.evaluation.EvaluationFramework import EvaluationFramework
from apsis.models.ParamInformation import NumericParamDef, NominalParamDef, \
    LowerUpperNumericParamDef, FixedValueParamDef
import os
import logging
import time

logging.basicConfig(level=logging.DEBUG)

def objective_func_from_sklearn(candidate, estimator, param_defs, X, y, parameter_names, scoring="accuracy", cv=3): #gets candidate, returns candidate.
    start_time = time.time()

    param_defs_sklearn = {}
    for i in range(len(parameter_names)):
        param_defs_sklearn[parameter_names[i]] = param_defs[i]

    print(str(param_defs_sklearn))

    sk_ad = SimpleScikitLearnAdapter(estimator, param_defs_sklearn)
    sk_learn_format = sk_ad.translate_vector_dict(candidate.params)
    print(str(sk_learn_format))

    estimator.set_params(**sk_learn_format)
    print(str(estimator.get_params()))
    scores = cross_val_score(estimator, X, y, scoring=scoring, cv=cv)
    print(str(scores))
    candidate.result = 1 - scores.mean()
    cost = time.time() - start_time
    candidate.cost = cost
    return candidate

#load mnist dataset
mnist = fetch_mldata('MNIST original',
                     data_home=os.environ.get('MNIST_DATA_CACHE', '~/.mnist-cache'))

print("Mnist Data Size " + str(mnist.data.shape))
print("Mnist Labels Size" + str(mnist.target.shape))

#train test split
mnist_data_train, mnist_data_test, mnist_target_train, mnist_target_test = \
    train_test_split(mnist.data, mnist.target, test_size=0.1, random_state=42)

regressor = SVC(kernel="poly")
parameter_names = ["C", "degree", "gamma", "coef0"]
param_defs = [
    LowerUpperNumericParamDef(0,1),
    FixedValueParamDef([1,2,3]),
    LowerUpperNumericParamDef(0, 1),
    LowerUpperNumericParamDef(0,1)
]

optimizer_args = {'minimization': True,
                  'initial_random_runs': 10}

"""sk_adapter = SimpleScikitLearnAdapter(regressor, param_defs,
                                              scoring="mean_squared_error",
                                              optimizer="SimpleBayesianOptimizationCore",
                                              optimizer_arguments=optimizer_args,n_iter=1)"""



obj_func_args = {"estimator": regressor,
                 "param_defs": param_defs,
                 "X": mnist_data_train,
                 "y": mnist_target_train,
                 "parameter_names": parameter_names}

ev = EvaluationFramework()
optimizers = [SimpleBayesianOptimizationCore({"param_defs": param_defs,
                                              "initial_random_runs": 10})]
steps = 20
ev.evaluate_optimizers(optimizers, ["BayOpt_MNIST_EI"],
                       objective_func_from_sklearn, objective_func_args=obj_func_args, obj_func_name="MNIST SVC",
                       steps=steps, show_plots_at_end=True)


"""



#scores = cross_val_score(regressor, mnist_data_train[:10], mnist_target_train[:10], scoring="mean_squared_error", cv=3)
#print(scores)

print(len(mnist_data_train))
print(len(mnist_target_train))

print(mnist_data_train[0])
print(mnist_target_train[0])
fitted = sk_adapter.fit(mnist_data_train[:10], mnist_target_train[:10])

print("----------------------------------\nTRAINING EVALUATION FOLLOWS\n----------------------------------")
print(mean_squared_error(mnist_target_train,
                                 fitted.predict(mnist_data_train)))
print("----------------------------------\nTest EVALUATION FOLLOWS\n----------------------------------")
print(mean_squared_error(mnist_target_test,
                                 fitted.predict(mnist_data_test)))
"""





