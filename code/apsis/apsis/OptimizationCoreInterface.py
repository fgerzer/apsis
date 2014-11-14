#!/usr/bin/python
from abc import ABCMeta, abstractmethod

from apsis.models.Candidate import Candidate

import logging


class OptimizationCoreInterface(object):
    """
    The interface definition all optimizers have to fulfill.

    Also stores a best_candidate.

    """
    __metaclass__ = ABCMeta

    SUPPORTED_PARAM_TYPES = None

    minimization = True
    param_defs = None

    best_candidate = None

    @abstractmethod
    def __init__(self, params):
        """
        Init method to initialize optimization core.

        Parameters
        ----------
        params: dict of string keys
            A dictionary of parameters for the optimizer. Should contain at
            least param_defs. For more arguments (which may be required by the
            individual optimizers) see their documentation.
        """
        pass

    @abstractmethod
    def next_candidate(self, worker_id=None):
        """
        This method is invoked by workers to obtain next candidate points that
        need to be evaluated.

        Parameters
        ----------
        worker_id: string or None
            A string id for the worker calling this method

        Returns
        -------
        next_candidate: Candidate
            A Candidate corresponding to the next point that shall be
            evaluated.
        """
        pass

    def working(self, candidate, status, worker_id=None, can_be_killed=False):
        """
        Method that is used by used by workers to announce their currently
        executed points.

        The working method is the main reply method for workers. It is used to
        announce to the optimizer that they are working on a point; to return
        points and to announce that they are paused. It also allows for an
        optimizer to kill workers.
        Later, workers may be required by a core to regularly send a working
        message.

        Parameters
        ----------
        candidate: Candidate
            Contains the information related to the point the worker is
            currently evaluating.

        status: {"finished", "working", "pausing"
            'finished' indicates that it stops the evaluation.
            'working' indicates that the worker wants to go on working
                on this point.
            'pausing' indicates that the worker needed to pause for some
                external reason and continuing on evaluating this point should
                be done by another worker.

        worker_id: string or None
            A string id for the worker calling this method. Unused by default.

        can_be_killed: bool
            Stating if the worker may be killed by the core.

        Returns
        -------
        continue: bool
            Tells the worker whether it should continue the computation.
        """
        #check for the new best result
        if self.best_candidate is not None:
            if self.is_better_candidate_as(candidate, self.best_candidate):
                logging.info("Cool - found new best candidate "
                             + str(candidate) + " with score "
                             + str(candidate.result) + " instead of "
                             + str(self.best_candidate.result))
                self.best_candidate = candidate

        else:
            self.best_candidate = candidate

    def is_better_candidate_as(self, one, two):
        """
        Tests whether one is a better Candidate than two.

        This is dependant on whether the problem is one of minimization or
        maximization.
        It is done by comparing their results.

        Parameters
        ----------
        one: Candidate
            Candidate that is tested on being better.

        two: Candidate
            Candidate that is used as a baseline.

        Returns
        -------
        better: bool
            True iff one is better than two

        Raises
        ------
        ValueError:
            If one or two are not Candidate objects.
        """
        if not isinstance(one, Candidate):
            raise ValueError("Value is not a candidate " + str(one)
                             + " but a " + str(type(one)))
        if not isinstance(two, Candidate):
            raise ValueError("Value is not a candidate " + str(two)
                             + " but a " + str(type(two)))

        if self.minimization:
            return one.result < two.result
        else:
            return one.result > two.result



    def _is_supported_param_type(self, param):
        """
        Tests whether a certain parameter is supported by the optimizer.

        Parameters
        ----------
        param:
            The parameter to be tested

        Result
        ------
        is_supported: bool
            True iff param is supported by this optimizer.
        """
        if isinstance(self.SUPPORTED_PARAM_TYPES, list):
            for sup in self.SUPPORTED_PARAM_TYPES:
                if isinstance(param, sup):
                    return True

        return False

    def _is_all_supported_param_types(self, param_list):
        """
        Tests whether a parameter list is completely supported by this
        optimizer.

        Parameters
        ----------
        param_list: list
            A list of parameter types.

        Returns
        -------

        is_supported: bool
            True iff all params in param_list are supported by this optimizer.
        """
        for param in param_list:
            if not self._is_supported_param_type(param):
                return False

        return True

class ListBasedCore(OptimizationCoreInterface):
    """"
    Defines a list-based core.

    A list-based core is characterized through the possession of three lists:
    finished_candidates, which stores all finished candidates
    working_candidates, which stores all candidates currently being worked on
    pending_candidates, which stores all candidates that have not yet been
        assigned.
    """
    finished_candidates = None
    working_candidates = None
    pending_candidates = None


    def __init__(self, params):
        """
        Init method to initialize the lists of ListBasedCore.

        Parameters
        ----------
        params: dict of string keys
            A dictionary of parameters for the optimizer. Should contain at
            least param_defs. For more arguments (which may be required by the
            individual optimizers) see their documentation. are ignored by
            this interface.
        """

        super(ListBasedCore, self).__init__(params)

        self.finished_candidates = []
        self.working_candidates = []
        self.pending_candidates = []



    def transfer_to_working(self, candidate):
        """
        Transfers a candidate to the working list if it isn't there already.

        Parameters
        ----------
        candidate: Candidate
            The candidate that should be checked.

        Returns
        -------
        continue: bool
            Returns True iff the candidate should be continued.
            Currently, the only way for it to return False is if the candidate
            has already been finished.
        """
        if candidate not in self.working_candidates:
            # work on a finished candidate is discarded and should abort
            if candidate in self.finished_candidates:
                return False
            #if work is carried out on candidate and it was pending it is no
            # longer pending
            elif candidate in self.pending_candidates:
                self.pending_candidates.remove(candidate)

            logging.debug("Candidate was UNKNOWN and not FINISHED "
                          + str(candidate)
                          + " Candidate added to WORKING list.")

            #but now it is a working item
            self.working_candidates.append(candidate)
        return True

    def deal_with_finished(self, candidate):
        """
        Deals with a finished candidate by checking for best results and doing
        list updates.

        Parameters
        ----------
        candidate: Candidate
            The candidate that has been finished.
        """
        self.working_candidates.remove(candidate)
        self.finished_candidates.append(candidate)

