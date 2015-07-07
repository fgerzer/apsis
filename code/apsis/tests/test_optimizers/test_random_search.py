__author__ = 'Frederik Diehl'

from apsis.optimizers.random_search import RandomSearch, QueueRandomSearch
from nose.tools import assert_is_none, assert_equal, assert_dict_equal, \
    assert_true, assert_false
from apsis.models.experiment import Experiment
from apsis.models.parameter_definition import MinMaxNumericParamDef, NominalParamDef
from apsis.models.candidate import Candidate
import multiprocessing
import os
import signal
import time

class test_RandomSearch(object):

    def test_init(self):
        #test initialization
        opt = RandomSearch(None)

    def test_get_next_candidate(self):
        opt = RandomSearch({"initial_random_runs": 3})
        exp = Experiment("test", {"x": MinMaxNumericParamDef(0, 1)}, NominalParamDef(["A", "B", "C"]))
        for i in range(5):
            cand = opt.get_next_candidates(exp)[0]
            assert_true(isinstance(cand, Candidate))
            cand.result = 2
            exp.add_finished(cand)
        cands = opt.get_next_candidates(exp, num_candidates=3)
        assert_equal(len(cands), 3)


class test_ParallelRandomSearch(object):

    def test_get_next_candidate(self):
        out_queue = multiprocessing.Queue()
        exp = Experiment("test", {"x": MinMaxNumericParamDef(0, 1)}, NominalParamDef(["A", "B", "C"]))
        opt = QueueRandomSearch({}, exp, out_queue)
        opt.start()
        for i in range(5):
            cand = out_queue.get()
            assert_true(isinstance(cand, Candidate))
            cand.result = 2
            exp.add_finished(cand)
        os.kill(opt.pid, signal.SIGINT)
        time.sleep(0.5)
        assert_false(opt.is_alive(), "Optimizer is still alive despite "
                                     "being killed.")