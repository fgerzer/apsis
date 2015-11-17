__author__ = 'Frederik Diehl'

from apsis.optimizers.optimizer import Optimizer, QueueBasedOptimizer, \
    QueueBackend
from apsis.models.experiment import Experiment
from apsis.models.parameter_definition import *
from nose.tools import assert_raises
from apsis.optimizers.random_search import RandomSearch
from multiprocessing import Queue
import time

class TestOptimizer(object):
    optimizer = None

    class OptimizerStub(Optimizer):
        SUPPORTED_PARAM_TYPES = [NumericParamDef]
        def get_next_candidates(self, num_candidates=1):
            pass

    def setup(self):
        param_def = {
            "x": MinMaxNumericParamDef(0, 1)
        }
        experiment = Experiment(name="test_optimizer_experiment",
                                parameter_definitions=param_def)
        self.optimizer = self.OptimizerStub(experiment, optimizer_params=None)

    def test_setup(self):
        pass

    def test_init_param_support(self):
        param_def = {
            "x": MinMaxNumericParamDef(0, 1),
            "not_supported": OrdinalParamDef(["A", "B"])
        }
        experiment = Experiment(name="test_optimizer_experiment_crash",
                                parameter_definitions=param_def)
        assert_raises(ValueError, self.OptimizerStub, experiment, None)

    def test_update_param_support(self):
        param_def = {
            "x": MinMaxNumericParamDef(0, 1),
            "not_supported": OrdinalParamDef(["A", "B"])
        }
        experiment = Experiment(name="test_optimizer_experiment_crash",
                                parameter_definitions=param_def)
        assert_raises(ValueError, self.optimizer.update, experiment)

class TestQueueOptimizer(object):
    optimizer = None

    def setup(self):
        param_def = {
            "x": MinMaxNumericParamDef(0, 1)
        }
        experiment = Experiment(name="test_optimizer_experiment",
                                parameter_definitions=param_def)
        self.optimizer = QueueBasedOptimizer(RandomSearch,
                                             experiment)

    def test_get_next_candidate(self):
        self.optimizer.get_next_candidates()

    def test_update(self):
        param_def = {
            "x": MinMaxNumericParamDef(0, 1)
        }
        experiment = Experiment(name="test_optimizer_experiment",
                                parameter_definitions=param_def)
        self.optimizer.update(experiment)

    def teardown(self):
        self.optimizer.exit()


class TestQueueBackend(object):
    backend = None
    experiment = None

    def setup(self):
        param_def = {
            "x": MinMaxNumericParamDef(0, 1)
        }
        self.experiment = Experiment(name="test_optimizer_experiment",
                                parameter_definitions=param_def)
        out_queue = Queue()
        in_queue = Queue()
        self.backend = QueueBackend(RandomSearch, self.experiment, out_queue,
                                    in_queue)

    def test_run(self):
        self.backend._in_queue.put("exit")
        self.backend.run()

    def test_check_update(self):
        self.backend._check_update()
        self.backend._in_queue.put("exit")
        time.sleep(0.1)
        self.backend._check_update()
        time.sleep(0.1)
        self.backend._check_update()

        self.backend._in_queue.put(self.experiment)
        time.sleep(0.1)
        self.backend._check_update()

    def test_check_generation(self):
        self.backend._check_generation()