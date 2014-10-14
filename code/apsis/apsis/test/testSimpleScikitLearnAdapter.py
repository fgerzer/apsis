__author__ = 'Frederik Diehl'

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from apsis.adapters.SimpleScikitLearnAdapter import SimpleScikitLearnAdapter
from sklearn.metrics import mean_squared_error
from apsis.models.ParamInformation import NumericParamDef, NominalParamDef


class testSimpleScikitLearnAdapter(object):
    def test_setup(self):
        boston_data = datasets.load_boston()
        regressor = LogisticRegression()

        param_defs = {
            "penalty": NominalParamDef(['l1', 'l2']),
            "C": NumericParamDef(0, 1)
        }
        sk_adapter = SimpleScikitLearnAdapter(regressor, param_defs,
                                              metric=mean_squared_error)

        fitted = sk_adapter.fit(boston_data.data, boston_data.target)

        print(mean_squared_error(boston_data.target,
                                 fitted.predict(boston_data.data)))