from apsis.utilities import adapter_utils
from sklearn.cross_validation import cross_val_score
from apsis.models.ParamInformation import ParamDef, NominalParamDef
import logging

class SimpleScikitLearnAdapter(object):
    """
    Simple Scikit Learn adaptor to be executed in a single core manner.
    """
    estimator = None
    param_defs = None
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

    def __init__(self, estimator, param_defs, n_iter=10, scoring=None, fit_params=None,
                 metric=None, n_jobs=1, refit=True, cv=None, random_state=None,
                 optimizer="RandomSearchCore", optimizer_arguments=None):
        self.estimator = estimator
        self.param_defs = param_defs
        self.n_iter = n_iter
        self.scoring = scoring
        self.fit_params = fit_params if fit_params is not None else {}
        self.metric = metric
        self.refit = refit
        self.cv = cv
        self.random_state = random_state
        self.optimizer_arguments = optimizer_arguments
        self.optimizer = optimizer
        self.worker_id = "SimpleScikitLearnAdapter-Worker"

    def translate_dict_vector(self, sklearn_params):
        """
        Helper method to translate from scikit learn hyperparam dictionaries
        to lists for this optimization framework


        :param sklearn_params: the dictionary of hyperparams as given by
        scikit learn's estimator.get_params()
        :return: a plain python list of the hyperparams
        """
        if self.parameter_names is None:
            self.parameter_names = []
            for k in sklearn_params.keys:
                self.parameter_names.append(k)

        converted_list = [None] * len(self.parameter_names)

        for i, name in enumerate(self.parameter_names):
            converted_list[i] = sklearn_params[name]

        return converted_list

    def translate_vector_dict(self, optimizer_params):
        """
        Translate back from the vector of hyperparams to the dictionary of
        hyperparams as used in scikit learn.
        First invokes translate_dict_vector to obtain dictionary keys.

        :param optimizer_params:
        :return: a dictionary compatible to the one used in self.estimator
        """

        #make sure to have the parameter names
        self.translate_dict_vector(self.estimator.get_params())

        return_dict = {}

        logging.debug("optimizer params %s, param names %s", str(optimizer_params), str(self.parameter_names))

        for i, name in enumerate(self.parameter_names):
            return_dict[name] = optimizer_params[i]

        return return_dict

    # noinspection PyPep8Naming
    def fit(self, X, y=None):
        """
        Method to run the optimizer bound to this adapter. Will optimize
        for n_iter steps using the OptimizationCoreInterface instance in
        optimizer.

        :param X: the unlabled data set of points
        :param y: the corresponding lables to the data set x
        :return: the original estimator if refit=False or the estimator
        refitted with the new found best hyper params
        when refit=True
        """

        #TODO convert param_defs to correct format
        optimizer_param_defs = self._convert_param_defs(self.param_defs)

        if self.optimizer_arguments is None:
            self.optimizer_arguments = {}
        self.optimizer_arguments['param_defs'] = optimizer_param_defs

        #use helper to find optimizer class and instantiate
        self.optimizer = adapter_utils.check_optimizer(self.optimizer)(
            self.optimizer_arguments)

        #now run the optimization for n_iter number of iterations
        for i in range(self.n_iter):
            #obtain a new candidate
            candidate = self.optimizer.next_candidate(self.worker_id)

            #convert candidate's hyperparam vector to sklearn format
            candidate_params_sklearn_format = self.translate_vector_dict(
                candidate.params)

            #build up estimator
            self.estimator.set_params(**candidate_params_sklearn_format)

            # noinspection PyPep8Naming

            #Get the scores in sklearn crossval notation.
            #scores is then a list of the results, and can be checked via
            #scores.mean() and scores.std().
            scores = cross_val_score(self.estimator, X, y,
                                     scoring=self.scoring, cv=self.cv)
            print("ScoresUIU: " +str(scores))
            candidate.result = scores.mean()

            #notify optimization core of completed evaluation result
            self.optimizer.working(candidate, "finished", self.worker_id)

        #store best parameters and corresponding result
        self.best_params = self.optimizer.best_candidate.params
        self.best_result = self.optimizer.best_candidate.result

        #check if estimator shall be refitted with new parameters.
        if self.refit:
            self.estimator.set_params(**self.translate_vector_dict(
                self.best_params))
            self.estimator.fit(X, y)
            logging.debug("Returning refitted estimator.")
            return self.estimator

        else:
            return self.estimator

    def _convert_param_defs(self, given_defs):
        param_list = []
        param_names = []
        for k in given_defs:
            if isinstance(given_defs[k], ParamDef):
                param_list.append(given_defs[k])
            elif isinstance(given_defs[k], list):
                param_list.append(NominalParamDef(given_defs[k]))
            else:
                raise ValueError("Parameter " + str(param_list[k])
                                 + " is not supported.")
            param_names.append(k)

        self.parameter_names = param_names
        return param_list



    def get_params(self):
        return self.best_params















