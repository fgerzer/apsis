from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from apsis.adapters.SimpleScikitLearnAdapter import SimpleScikitLearnAdapter
from sklearn.metrics import mean_squared_error
from apsis.models.ParamInformation import NumericParamDef, NominalParamDef, \
    LowerUpperNumericParamDef


class testSimpleBayesianOptimizationCore(object):

    def test_setup(self):
        boston_data = datasets.load_boston()
        regressor = LogisticRegression()

        param_defs = {
            "C": LowerUpperNumericParamDef(0, 1)
        }

        sk_adapter = SimpleScikitLearnAdapter(regressor, param_defs,
                                              scoring="mean_squared_error",
                                              n_iter=20,
                                              optimizer='SimpleBayesianOptimizationCore',
                                              optimizer_arguments={'initial_random_runs': 3}
                                              )

        fitted = sk_adapter.fit(boston_data.data, boston_data.target)

        print(mean_squared_error(boston_data.target,
                                 fitted.predict(boston_data.data)))
