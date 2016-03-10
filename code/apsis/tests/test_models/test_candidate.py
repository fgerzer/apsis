__author__ = 'Frederik Diehl'

from apsis.models.candidate import Candidate, from_dict
from nose.tools import assert_dict_equal, assert_equal, assert_raises, \
    assert_not_equal, assert_false, assert_true


class TestCandidate(object):
    """
    Tests the candidate.
    """

    def test_init(self):
        """
        Tests the initialization.
            - Raising ValueError when no dict is given
            - Parameter correctness
            - worker_information correctness
        """
        params = {
            "x": 1,
            "name": "B"
        }
        worker_info = "test_worker_info."
        cand1 = Candidate(params, worker_information=worker_info)
        assert_dict_equal(cand1.params, params)
        assert_equal(cand1.worker_information, worker_info)

        with assert_raises(ValueError):
            Candidate(False)

    def test_eq(self):
        """
        Tests the equiality.
            - Equal works
            - Equal with a non-Candidate works
        """
        params = {
            "x": 1,
            "name": "B"
        }
        params2 = {
            "x": 2,
            "name": "B"
        }
        cand1 = Candidate(params)
        cand2 = Candidate(params)

        assert_not_equal(cand1, cand2)
        cand3 = Candidate(params2, cand_id=cand1.cand_id)
        assert_true(cand1.__eq__(cand3))

        assert_false(cand1.__eq__(False))

    def test_str(self):
        """
        Tests whether stringify works.
        """
        params = {
            "x": 1,
            "name": "B"
        }
        cand1 = Candidate(params)
        cand1.cost = 2
        str(cand1)

    def test_dict(self):
        """
        Tests the to-dict and from-dict methods.
        """
        params = {
            "x": 1,
            "name": "B"
        }
        cand1 = Candidate(params)
        entry = cand1.to_dict()
        d = {"params": params,
             "result": None,
             "cost": None,
             "worker_information": None,
             "failed": False,
             "cand_id": cand1.cand_id}
        assert_dict_equal(entry, d)

        cand2 = from_dict(entry)
        assert_equal(cand1, cand2)