__author__ = 'Frederik Diehl'

from apsis.utilities.optimizer_utils import *
from apsis.models.experiment import Experiment
from apsis.models.parameter_definition import *
from nose.tools import assert_is_instance, assert_raises, assert_equal
import time

class TestOptimizerUtils(object):

    def test_check_optimizer(self):
        param_def = {
            "x": MinMaxNumericParamDef(0, 1)
        }
        experiment = Experiment(name="test_optimizer_experiment",
                                parameter_definitions=param_def)
        assert_is_instance(check_optimizer(RandomSearch, experiment, {"multiprocessing": "none"}), RandomSearch)
        assert_is_instance(check_optimizer("RandomSearch", experiment, {"multiprocessing": "none"}), RandomSearch)

        with assert_raises(ValueError):
            check_optimizer("fails", experiment, {"multiprocessing": "none"})
        with assert_raises(ValueError):
            check_optimizer(MinMaxNumericParamDef, experiment)

        queue_based = check_optimizer(RandomSearch, experiment,
                                           {"multiprocessing": "queue"})
        assert_is_instance(queue_based, QueueBasedOptimizer)
        assert_equal(check_optimizer(queue_based, experiment,
                       {"multiprocessing": "queue"}), queue_based)
        queue_based.exit()
        assert_is_instance(check_optimizer(RandomSearch, experiment,
                                           {"multiprocessing": "none"}),
                           RandomSearch)
        with assert_raises(ValueError):
            check_optimizer(RandomSearch, experiment,
                                               {"multiprocessing": "fails"}),

        time.sleep(0.1)