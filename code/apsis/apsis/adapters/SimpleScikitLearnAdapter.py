from apsis import OptimizationCoreInterface, RandomSearchCore
from apsis.OptimizationCoreInterface import OptimizationCoreInterface
from apsis.Candidate import Candidate
from apsis.helpers import adapter_helpers
import numpy as np
from sklearn.cross_validation import train_test_split

class SimpleScikitLearnAdapter:
    estimator = None
    n_iter = None
    scoring = None
    fit_params = None
    n_jobs = None
    refit = None
    cv = None
    random_state = None
    optimizer = None
    optimizer_arguments = None
    worker_id = None
    parameter_names = None
    metric = None

    best_params = None
    best_result = None

    def __init__(self, estimator, n_iter=10, scoring=None, fit_params=None, metric=None, n_jobs=1, refit=True, cv=None, random_state=None, optimizer="RandomSearchCore", optimizer_arguments=None):
        self.estimator = estimator
        self.n_iter = n_iter
        self.scoring = scoring
        self.fit_params = fit_params
        self.metric = metric
        self.refit = refit
        self.cv = cv
        self.random_state = random_state
        self.optimizer_arguments = optimizer_arguments
        self.optimizer = optimizer
        self.worker_id = "SimpleScikitLearnAdapter-Worker"

    def translate_dict_vector(self, sklearn_params):
        if self.parameter_names is None:
            self.parameter_names = []
            for k in sklearn_params.keys:
                self.parameter_names.append(k)

        return_vector = np.zeros(len(self.parameter_names), 1)

        for i, name in enumerate(self.parameter_names):
            return_vector[i] = sklearn_params[name]

        return return_vector

    def translate_vector_dict(self, optimizer_params):
        return_dict = {}

        for i, name in enumerate(self.parameter_names):
            return_dict[name] = optimizer_params[i]

        return return_dict


    def fit(self, X, y=None):
        #use helper to find optimizer class and instantiate
        self.optimizer = adapter_helpers.check_optimizer(self.optimizer)(self.optimizer_arguments)

        #make sure to have the parameter names
        self.translate_vector_dict(self.estimator.get_params())

        for i in range(self.n_iter):
            candidate = self.optimizer.next_candidate("SimpleScikitLearnAdapter-Worker")

            candidate_params_sklearn_format = self.translate_vector_dict(candidate.params)

            self.estimator.set_params(candidate_params_sklearn_format)

            #TODO use CV
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

            self.estimator.fit(X_train, y_train)

            y_pred = self.estimator.predict(X_test)

            result = self.metric(y_test, y_pred)

            candidate.result = result

            self.optimizer.working(candidate, "finished", self.worker_id)


        self.best_params = self.optimizer.best_candidate.params
        self.best_result = self.optimizer.best_candidate.result

        if self.refit:
            self.estimator.set_params(self.best_params)
            self.estimator.fit(X, y)
            return self.estimator

        else:
            return self.estimator


    def get_params(self):
        return self.best_params















