#!/usr/bin/python

from apsis.OptimizationCoreInterface import OptimizationCoreInterface
from apsis.Candidate import Candidate
from sklearn.utils import check_random_state
import numpy as np
import logging


class RandomSearchCore(OptimizationCoreInterface):
    lower_bound = None
    upper_bound = None
    random_state = None

    finished_candidates = []
    working_candidates = []
    pending_candidates = []

    best_candidate = None


    def __init__(self, lower_bound, upper_bound, minimization_problem=True, random_state=0):
        print("Initializing Random Search Core for bounds..." + str(lower_bound) + " and " + str(upper_bound))

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.random_state = check_random_state(random_state)
        self.minimization = minimization_problem

    #TODO deal with the case that candidate point is the same but objects do not equal
    def working(self, candidate, status, worker_id=None, can_be_killed=False):
        logging.debug("Worker " + str(worker_id) + " informed me about work in status " + str(status) + "on candidate " + str(candidate))

        #first check if this point is known
        if candidate not in self.working_candidates:
            #work on a finished candidate is discarded and should abort
            if candidate in self.finished_candates:
                return False
            #if work is carried out on candidate and it was pending it is no longer pending
            elif candidate in self.pending_candidates:
                self.pending_candidate.remove(candidate)

            logging.debug("Candidate was UNKNOWN and not FINISHED " + str(candidate) + " Candidate added to WORKING list.")

            #but now it is a working item
            self.working_candidates.append(candidate)

        #if finished remove from working and put to finished list.
        if status == "finished":
            self.working_candidates.remove(candidate)
            self.finished_candidates.append(candidate)

            #check for the new best result
            if self.best_candidate is not None:
                if self.is_better_candidate_as(candidate, self.best_candidate):
                    logging.info("Cool - found new best candidate " + str(candidate))
                    self.best_candidate = candidate

            else:
                self.best_candidate = candidate

            return False
        elif status == "working":
            #for now just continue working
            return True

        elif status == "pausing":
            self.working_candidates.remove(candidate)
            self.pending_candidates.append(candidate)

            return False

        else:
            logging.error("Worker " + worker_id + " posted candidate to core with non correct status value " + status)

        return True

    def next_candidate(self, worker_id=None):
        """
        This method is invoked by workers to obtain next candidate points that need to be evaluated.

        The new Candidate object is return following these rules:
        If pending candidates are there, they are returned first. Otherwise a new random candidate is generated at
        uniform. It is made sure that this point has not been marked as finished yet. On return of a new candidate
        it is appended to the pending_candidates list in this core.

        :param worker_id: not used here, but needed to satisfy interface of OptimizationCoreInterface.
        :return: an instance of a Candidate object that should be evaluated next.
        """
        #either we have pending candidates
        if len(self.pending_candidates) > 0:
            new_candidate = self.pending_candidates.pop(0)

            logging.debug("Core providing pending candidate " + new_candidate)

        #or we need to generate new ones
        else:
            new_candidate_point = self.generate_new_random_vec()
            new_candidate = Candidate(new_candidate_point)

            while (new_candidate in self.finished_candidates) or (new_candidate in self.working_candidates) or (new_candidate in self.pending_candidates):
                new_candidate_point = self.generate_new_random_vec()
                new_candidate = Candidate(new_candidate_point)

            logging.debug("Core generated new point to evaluate " + str(new_candidate))

        #add candidate to working list
        self.working_candidates.append(new_candidate)

        return new_candidate

    def generate_new_random_vec(self):
        new_candidate_point = np.zeros(self.lower_bound.shape)

        for i in range(new_candidate_point.shape[0]):
            new_candidate_point[i] = self.random_state.uniform(self.lower_bound[i], self.upper_bound[i])

        return new_candidate_point

