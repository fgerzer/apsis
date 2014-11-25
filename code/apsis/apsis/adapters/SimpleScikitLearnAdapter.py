from apsis.utilities import adapter_utils
from sklearn.cross_validation import cross_val_score
from apsis.models.ParamInformation import ParamDef, NominalParamDef
import logging


class SimpleScikitLearnAdapter(object):
    """
    A simple scikit-learn adapter for (currently) single-core optimization.

    Functionality is simple, and implements scikit-learn's estimator interface.
    In general, using this adapter works by first instantiating it with the
    details of the optimization - the optimizer, arguments for it, whether to
    use crossvalidation, which scoring to use for example - then calling the
    fit method.
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

    best_params = None
    best_result = None

    def __init__(self, estimator, param_defs, n_iter=10, scoring=None,
                 fit_params=None, n_jobs=1, refit=True, cv=3,
                 random_state=None, optimizer="RandomSearchCore",
                 optimizer_arguments=None):
        """
        Initializes the SimpleScikitLearnAdapter.

        Parameters
        ----------
        estimator: scikit learn baseEstimator
            The estimator for which the parameters should be optimized. Needs
            a set_param, get_param and fit function.

        param_defs: List of models.ParamInformation
            Defines the parameter ranges as used by the estimator. Different
            optimizers may support different ParamInformations.

        n_iter: int
            The number of iterations the optimizer may run.

        scoring: scikit-learn scoring object
            Defines the way the best result is computed.

        fit_params: dict of string keys
            Which parameters to give to the estimator.fit method.

        n_jobs: int
            CURRENTLY UNUSED
            Defines how many threads may run concurrently.

        refit: bool
            Whether to return a refitted estimator on the whole dataset, or
            not.

        cv: int
            Number of crossvalidations to use.

        random_state: RandomState object or None
            See
            http://scikit-learn.org/stable/developers/index.html#random-numbers

        optimizer: string or OptimizationCoreInterface object.
            Defines the optimizer which is used to optimize the
            hyper-parameters.

        optimizer_arguments: dict of string keys
            Arguments for the optimizer.
        """
        self.estimator = estimator
        self.param_defs = param_defs
        self.n_iter = n_iter
        self.parameter_names = self.param_defs.keys()
        self.scoring = scoring
        self.fit_params = fit_params if fit_params is not None else {}
        self.refit = refit
        self.cv = cv
        self.random_state = random_state
        self.optimizer_arguments = optimizer_arguments
        self.optimizer = optimizer
        self.worker_id = "SimpleScikitLearnAdapter-Worker"

    def translate_dict_vector(self, sklearn_params):
        """
        Helper method to translate from scikit learn hyperparam dictionaries
        to lists for this optimization framework.

        Parameters
        ----------

        sklearn_params: dict of strings
            the dictionary of hyperparams as given by scikit learn's
            estimator.get_params()

        Returns
        -------
        converted_list: array_like
            A plain python list of the hyperparam values
        """

        #If we do not yet know which parameter names exist, we fill our
        #self.parameter_names list here.

        converted_list = [None] * len(self.parameter_names)

        for i, name in enumerate(self.parameter_names):
            converted_list[i] = sklearn_params[name]

        return converted_list

    def translate_vector_dict(self, optimizer_params):
        """
        Translate back from the vector of hyperparams to the dictionary of
        hyperparams as used in scikit learn.
        First invokes translate_dict_vector to obtain dictionary keys.

        Parameters
        ----------

        optimizer_params: array_like
            A plain python list of the hyperparam values

        Returns
        -------
        return_dict: dict of string keys
            A dictionary compatible to the one used in
            self.estimator.get_params
        """

        # make sure to have the parameter names
        self.translate_dict_vector(self.estimator.get_params())

        return_dict = {}

        logging.debug("optimizer params %s, param names %s",
                      str(optimizer_params), str(self.parameter_names))
        for i, name in enumerate(self.parameter_names):
            return_dict[name] = optimizer_params[i]

        return return_dict

    def fit(self, X, y=None, warm_start=False):
        """
        Method to run the optimizer bound to this adapter. Will optimize
        for self.n_iter steps using the OptimizationCoreInterface instance in
        self.optimizer.

        Parameters
        ----------
        X: nd_array
            The unlabeled data set of points

        y: nd_array
            The corresponding labels of the data set X.

        warm_start: bool
            If true, uses knowledge of the last run.

        Returns
        -------
        estimator: scikitlearn's baseEstimator
            the original estimator if refit=False or the estimator
        refitted with the new found best hyper params if refit=True
        """

        optimizer_param_defs = self._convert_param_defs(self.param_defs)

        if self.optimizer_arguments is None:
            self.optimizer_arguments = {}
        self.optimizer_arguments['param_defs'] = optimizer_param_defs

        #use helper to find optimizer class and instantiate
        if not warm_start:
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

            #Get the scores in sklearn crossval notation.
            #scores is then a list of the results, and can be checked via
            #scores.mean() and scores.std().
            scores = cross_val_score(self.estimator, X, y,
                                     scoring=self.scoring, cv=self.cv)

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

    def _convert_param_defs(self, given_defs):
        """
        Makes sure that the given_defs are in ParamDef format.

        Parameters
        ----------
        given_defs: list
            The parameter definitions. For each of them, they are either left
            through directly (iff ParamInformation), converted to
            NominalParamDef (iff list) or a ValueError is raised.

        Returns
        -------
        param_list: list of ParamInformation
            The completely converted list of ParamInformations.

        Raises
        ------
        ValueError:
            If one of the entries is neither ParamInformation or list.
        """
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
        """
        Returns the currently best parameters.

        Returns
        -------
        best_params: list
            The best parameters as currently found.

        """
        return self.best_params