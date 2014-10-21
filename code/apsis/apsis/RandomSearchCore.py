#!/usr/bin/python

import logging

from sklearn.utils import check_random_state
import numpy as np
import random

from apsis.OptimizationCoreInterface import OptimizationCoreInterface, ListBasedCore
from apsis.models.Candidate import Candidate
from apsis.utilities.validation import check_array, \
    check_array_dimensions_equal
from apsis.models.ParamInformation import ParamDef, NominalParamDef, NumericParamDef


class RandomSearchCore(OptimizationCoreInterface, ListBasedCore):
    """
    Implements a random searcher for parameter optimization.
    """
    random_state = None

    SUPPORTED_PARAM_TYPES = [NominalParamDef, NumericParamDef]

    def __init__(self, params):
        """
        Initializes the random search.


        :raise ValueError: If params does not contain lower_bound or
        upper_bound, or attributes are assigned bad values.
        """
        if params is None:
            raise ValueError("No params dict given!")

        if params.get('param_defs', None) is None:
            raise ValueError("Parameter definition list is missing!")

        #check if param_defs are supported
        if not self._is_all_supported_param_types(params["param_defs"]):
            raise ValueError(
                "Param list contains parameters of unsopported types. "
                "Supported types are  " + str(self.SUPPORTED_PARAM_TYPES))

        self.param_defs = params["param_defs"]

        logging.debug("Initializing Random Search Core for bounds...")

        self.random_state = check_random_state(
            params.get("random_state", None))
        self.minimization = params.get("minimization_problem", True)

        self.finished_candidates = []
        self.working_candidates = []
        self.pending_candidates = []

        super(RandomSearchCore, self).__init__(params)

    # TODO deal with the case that candidate point is the same but
    # objects do not equal
    def working(self, candidate, status, worker_id=None, can_be_killed=False):
        """
        Right now, RandomSearchCore works like this:
        It ensures candidate is in the working_candidates list. If it is in
        the finished_candidates list,
         the worker is told to terminate execution.
        If the status is 'working', the worker may continue.
        If the status is 'pausing', candidate is returned to the
        pending_candidates list.
        If the status is 'finished', candidate is appended to the
        finished_candidates list, and possibly the best
         result updated. Of course, the worker is then told to stop.
        """
        logging.debug("Worker " + str(worker_id) + " informed me about work "
                                                   "in status " + str(status)
                      + "on candidate " + str(candidate))

        # first check if this point is known
        self.perform_candidate_state_check(candidate)

        # if finished remove from working and put to finished list.
        if status == "finished":
            self.deal_with_finished(candidate)

            return False
        elif status == "working":
            #for now just continue working
            return True

        elif status == "pausing":
            self.working_candidates.remove(candidate)
            self.pending_candidates.append(candidate)

            return False

        else:
            logging.error("Worker " + worker_id + " posted candidate to core "
                                                  "with non correct status "
                                                  "value " + status)

        return True

    def next_candidate(self, worker_id=None):
        """
        This method is invoked by workers to obtain next candidate points
        that need to be evaluated.

        The new Candidate object is return following these rules:
        If there are pending candidates, they are returned first. Otherwise
        a new random candidate is generated
        from a uniform distribution. It is made sure that this point has not
        been marked as finished yet.
        On return of a new candidate it is appended to the working_candidates
        list in this core.

        :param worker_id: not used here, but needed to satisfy interface of
        OptimizationCoreInterface.
        :return: an instance of a Candidate object that should be evaluated
        next.
        """
        # either we have pending candidates
        if len(self.pending_candidates) > 0:
            new_candidate = self.pending_candidates.pop(0)

            logging.debug("Core providing pending candidate "
                          + str(new_candidate))

        # or we need to generate new ones
        else:
            new_candidate_point = self._generate_new_random_vector()
            new_candidate = Candidate(new_candidate_point)

            #Otherwise, a nominal-data-only parameter distribution might
            #run out of possible parameter combinations.
            total_possible_param_comb = 1
            for p in self.param_defs:
                if isinstance(p, NominalParamDef):
                    total_possible_param_comb *= len(p.values)
                elif isinstance(p, NumericParamDef):
                    total_possible_param_comb = float("inf")

            if (len(self.finished_candidates) + len(self.working_candidates)
                    + len(self.pending_candidates) < total_possible_param_comb):
                while ((new_candidate in self.finished_candidates)
                       or (new_candidate in self.working_candidates)
                    or (new_candidate in self.pending_candidates)):
                    new_candidate_point = self._generate_new_random_vector()
                    new_candidate = Candidate(new_candidate_point)
            else:
                logging.info("Core generated more than "
                             + str(total_possible_param_comb) + " points.")

            logging.debug("Core generated new point to evaluate " +
                          str(new_candidate))

        # add candidate to working list
        self.working_candidates.append(new_candidate)

        return new_candidate

    def _generate_new_random_vector(self):
        """
        Generates a new random vector. Hast to take care of parameter type
        """
        # initialize empty list of correct length
        new_candidate_point = [None] * len(self.param_defs)

        for i in range(len(new_candidate_point)):
            param_information = self.param_defs[i]

            if isinstance(param_information, NumericParamDef):
                new_candidate_point[i] = param_information.warp_out(
                    self.random_state.uniform(0, 1))

            elif isinstance(param_information, NominalParamDef):
                new_candidate_point[i] = random.choice(param_information.values)

        return new_candidate_point

