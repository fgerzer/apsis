__author__ = 'Frederik Diehl'

from apsis.models.candidate import Candidate
from apsis.models.parameter_definition import ParamDef
import copy
import uuid


class Experiment(object):
    """
    An Experiment is a set of parameter definitions and multiple candidate
    evaluations thereon.

    Attributes
    ----------
    name : string
        The name of the experiment. This does not have to be unique, but is
        for human orientation.
    parameter_definitions : dict of ParamDefs
        A dictionary of ParamDef instances. These define the parameter space
        over which optimization is possible.
    minimization_problem : bool
        Defines whether the experiment's goal is to find a minimum result - for
        example when evaluating errors - or a maximum result - for example when
        evaluating scores.
    candidates_pending : list of Candidate instances
        These Candidate instances have been generated by an optimizer to be
        evaluated at the next possible time, but are not yet assigned to a
        worker.
    candidates_working : list of Candidate instances
        These Candidate instances are currently being evaluated by workers.
    candidates_finished : list of Candidate instances
        These Candidate instances have finished evaluated.
    best_candidate : Candidate instance
        The as of yet best Candidate instance found, according to the result.
    note : string, optional
        The note can be used to add additional human-readable information to
        the experiment.
    """
    name = None

    parameter_definitions = None
    minimization_problem = None

    notes = None
    exp_id = None

    candidates_pending = None
    candidates_working = None
    candidates_finished = None

    best_candidate = None

    def __init__(self, name, parameter_definitions, exp_id=None, notes=None, minimization_problem=True,):
        """
        Initializes an Experiment with a certain parameter definition.

        All of the Candidate lists are set to empty lists, representing an
        experiment with no work done.

        Parameters
        ----------
        name : string
            The name of the experiment. This does not have to be unique, but is
            for human orientation.

        parameter_definitions : dict of ParamDef
            Defines the parameter space of the experiment. Each entry of the
            dictionary has to be a ParamDef, and it is that space over which
            optimization will occur.
        minimization_problem : bool, optional
            Defines whether the experiment's goal is to find a minimum result - for
            example when evaluating errors - or a maximum result - for example when
            evaluating scores. Is True by default.
        notes : string or None, optional
            The note can be used to add additional human-readable information to
            the experiment.
        Raises
        ------
        ValueError :
            Iff parameter_definitions are not a dictionary.
        """
        self.name = name
        if exp_id is None:
            exp_id = uuid.uuid4().hex
        self.exp_id = exp_id
        if not isinstance(parameter_definitions, dict):
            raise ValueError("parameter_definitions are not a dict.")
        for p in parameter_definitions:
            if not isinstance(parameter_definitions[p], ParamDef):
                raise ValueError("Parameter definition of %s is not a ParamDef."
                                 %p)
        self.parameter_definitions = parameter_definitions

        self.minimization_problem = minimization_problem

        self.candidates_finished = []
        self.candidates_pending = []
        self.candidates_working = []

        self.notes = notes

    def add_finished(self, candidate):
        """
        Announces a Candidate instance to be finished evaluating.

        This moves the Candidate instance to the candidates_finished list and
        updates the best_candidate.

        Parameters
        ----------
        candidate : Candidate
            The Candidate to be added to the finished list.

        Raises
        ------
        ValueError :
            Iff candidate is not a Candidate object.
        """
        if not isinstance(candidate, Candidate):
            raise ValueError("candidate is not an instance of Candidate.")
        if not self._check_candidate(candidate):
            raise ValueError("candidate %s is not valid." %candidate)
        if candidate in self.candidates_pending:
            self.candidates_pending.remove(candidate)
        if candidate in self.candidates_working:
            self.candidates_working.remove(candidate)
        if candidate in self.candidates_finished:
            self.candidates_finished.remove(candidate)
        self.candidates_finished.append(candidate)
        self._update_best()

    def add_pending(self, candidate):
        """
        Adds a new pending Candidate object to be evaluated.

        This function should be used when a new pending candidate is supposed
        to be evaluated. If an old Candidate should be updated as just pausing,
        use the add_pausing function.

        Parameters
        ----------
        candidate : Candidate
            The Candidate instance that is supposed to be evaluated soon.

        Raises
        ------
        ValueError :
            Iff candidate is no Candidate object.
        """
        if not isinstance(candidate, Candidate):
            raise ValueError("candidate is not an instance of Candidate.")
        if not self._check_candidate(candidate):
            raise ValueError("candidate is not valid.")
        if candidate in self.candidates_pending:
            self.candidates_pending.remove(candidate)
        if candidate in self.candidates_working:
            self.candidates_working.remove(candidate)
        if candidate in self.candidates_finished:
            self.candidates_finished.remove(candidate)
        self.candidates_pending.append(candidate)
        self._update_best()

    def add_working(self, candidate):
        """
        Updates the experiment to now start working on candidate.

        This updates candidates_working list and the candidates_pending list
        if candidate is in the candidates_pending list.

        Parameters
        ----------
        candidate : Candidate
            The Candidate instance that is currently being worked on.

        Raises
        ------
        ValueError :
            Iff candidate is no Candidate object.
        """
        if not isinstance(candidate, Candidate):
            raise ValueError("candidate is not an instance of Candidate.")
        if not self._check_candidate(candidate):
            raise ValueError("candidate is not valid.")
        if candidate in self.candidates_pending:
            self.candidates_pending.remove(candidate)
        if candidate in self.candidates_working:
            self.candidates_working.remove(candidate)
        if candidate in self.candidates_finished:
            self.candidates_finished.remove(candidate)
        self.candidates_working.append(candidate)
        self._update_best()

    def add_pausing(self, candidate):
        """
        Updates the experiment that work on candidate has been paused.

        This updates candidates_pending list and the candidates_working list
        if it contains the candidate.

        Parameters
        ----------
        candidate : Candidate
            The Candidate instance that is currently paused.

        Raises
        ------
        ValueError :
            Iff candidate is no Candidate object.

        """
        if not isinstance(candidate, Candidate):
            raise ValueError("candidate is not an instance of Candidate.")
        if not self._check_candidate(candidate):
            raise ValueError("candidate is not valid.")
        if candidate in self.candidates_working:
            self.candidates_working.remove(candidate)
        if candidate in self.candidates_pending:
            self.candidates_pending.remove(candidate)
        if candidate in self.candidates_working:
            self.candidates_working.remove(candidate)
        if candidate in self.candidates_finished:
            self.candidates_finished.remove(candidate)
        self.candidates_pending.append(candidate)
        self._update_best()

    def better_cand(self, candidateA, candidateB):
        """
        Determines whether CandidateA is better than candidateB in the context
        of this experiment.
        This is done as follows:
        If candidateA's result is None, it is not better.
        If candidateB's result is None, it is better.
        If it is a minimization problem and the result is smaller than B's, it
        is better. Corresponding for being a maximization problem.


        Parameters
        ----------
        candidateA : Candidate
            The candidate which should be better.
        candidateB : Candidate
            The baseline candidate.

        Returns
        -------
        result : bool
            True iff A is better than B.

        Raises
        ------
        ValueError :
            If candidateA or candidateB are no Candidates.
        """
        if not isinstance(candidateA, Candidate) and candidateA is not None:
            raise ValueError("candidateA is %s, but no Candidate instance."
                             %str(candidateA))
        if not isinstance(candidateB, Candidate) and candidateB is not None:
            raise ValueError("candidateB is %s, but no Candidate instance."
                             %str(candidateB))

        if candidateA is None:
            return False
        if candidateB is None:
            return True

        if not self._check_candidate(candidateA):
            raise ValueError("candidateA is not valid.")
        if not self._check_candidate(candidateB):
            raise ValueError("candidateB is not valid.")

        a_result = candidateA.result
        b_result = candidateB.result

        if a_result is None:
            return False
        if b_result is None:
            return True
        if self.minimization_problem:
            if a_result < b_result:
                return True
            else:
                return False
        else:
            if a_result > b_result:
                return True
            else:
                return False

    def warp_pt_in(self, params):
        """
        Warps in a point.

        Parameters
        ----------
        params : dict of string keys
            The point to warp in

        Returns
        -------
        warped_in : dict of string keys
            The warped-in parameters.
        """
        warped_in = {}
        for name, value in params.iteritems():
            warped_in[name] = self.parameter_definitions[name].warp_in(value)
        return warped_in

    def warp_pt_out(self, params):
        """
        Warps out a point.

        Parameters
        ----------
        params : dict of string keys
            The point to warp out

        Returns
        -------
        warped_out : dict of string keys
            The warped-out parameters.
        """
        warped_out = {}
        for name, value in params.iteritems():
            warped_out[name] = self.parameter_definitions[name].warp_out(value)
        return warped_out

    def to_csv_results(self, delimiter=",", line_delimiter="\n", key_order=None, wHeader=True, fromIndex=0):
        """
        Generates a csv result string from this experiment.

        Parameters
        ----------
            delimiter : string, optional
                The column delimiter
            line_delimiter : string, optional
                The line delimiter
            key_order : list of strings, optional
                The order in which the parameters should be written. If None,
                the order is defined by sorting the parameter names.
            wHeader : bool, optional
                Whether a header should be written. Defaults to true.
            from_index : int, optional
                Beginning from which result the csv should be generated.

        Returns
        -------
            csv_string : string
                The corresponding csv string
            steps_included : int
                The number of steps included in the csv.
        """
        csv_string = ""
        if key_order is None:
            key_order = sorted(self.parameter_definitions.keys())

        if wHeader:
            csv_string += "step" + delimiter + "cand_id" + delimiter
            for k in key_order:
                csv_string += k + delimiter
            csv_string += "cost" + delimiter + "result" + delimiter + \
                          "best_result" + line_delimiter

        steps_included = 0
        for c in range(fromIndex, len(self.candidates_finished)):
            cand = self.candidates_finished[c]
            csv_string += str(c + 1) + delimiter + \
                          cand.to_csv_entry(delimiter=delimiter,key_order=key_order) \
                          + delimiter + str(self.best_candidate.result) + line_delimiter

            steps_included += 1

        return csv_string, steps_included

    def clone(self):
        """
        Create a deep copy of this experiment and return it.

        Returns
        -------
            copied_experiment : Experiment
                A deep copy of this experiment.
        """
        copied_experiment = copy.deepcopy(self)

        return copied_experiment

    def _check_candidate(self, candidate):
        """
        Checks whether candidate is valid for this experiment.

        This checks the existence of all parameter definitions and that all
        values are acceptable.

        Parameter
        ---------
        candidate : Candidate
            Candidate to check

        Returns
        -------
        acceptable : bool
            True iff the candidate is valid
        """
        if not set(candidate.params.keys()) == set(self.parameter_definitions.keys()):
            return False

        for k in candidate.params:
            if not self.parameter_definitions[k].is_in_parameter_domain(candidate.params[k]):
                return False
        return True

    def _check_param_dict(self, param_dict):
        """
        Checks whether parameter dictionary is valid for this experiment.

        This checks the existence of all parameter definitions and that all
        values are acceptable.

        Parameter
        ---------
        param_dict : dict with string keys
            Dictionary to check

        Returns
        -------
        acceptable : bool
            True iff the dictionary is valid
        """
        if not set(param_dict.keys()) == set(self.parameter_definitions.keys()):
            return False

        for k in param_dict:
            if not self.parameter_definitions[k].is_in_parameter_domain(param_dict[k]):
                return False
        return True

    def to_dict(self):
        """
        Generates a dictionary describing the current state of the experiment.

        This dictionary contains the following keys/value pairs:
            - "name": The name of the experiment
            - "parameter_definition": The parameter definition. Contains, for
                each parameter, a key/value pair of the parameter name and the
                dictionary as defined by param_def.to_dict().
            - "mimization_problem": Boolean. If true, the problem is one we
                want to minimize. If false, it's one we want to maximize.
            - "notes": Notes the user has entered for the experiment.
            - "exp_id": The ID of the experiment.
            - One list each for candidates_finished, -_pending and -_working.
                Contains, for each candidate in the respective list, a
                dictionary as defined by Candidate.to_dict().
            - "best_candidate": The best candidate or None.

        Returns
        -------
            dict : dict
                Dictionary as defined above.

        """
        param_defs = {}
        for k in self.parameter_definitions:
            param_defs[k] = self.parameter_definitions[k].to_dict()

        cand_dict_finished = [c.to_dict() for c in self.candidates_finished]
        cand_dict_pending = [c.to_dict() for c in self.candidates_pending]
        cand_dict_working = [c.to_dict() for c in self.candidates_working]

        result_dict = {"name": self.name,
                "parameter_definitions": param_defs,
                "minimization_problem": self.minimization_problem,
                "notes": self.notes,
                "exp_id": self.exp_id,
                "candidates_finished": cand_dict_finished,
                "candidates_pending": cand_dict_pending,
                "candidates_working": cand_dict_working
                }
        if self.best_candidate is not None:
            result_dict["best_candidate"] = self.best_candidate.to_dict()
        else:
            result_dict["best_candidate"] = None
        return result_dict

    def _update_best(self):
        best_candidate = None
        for c in self.candidates_finished:
            if self.better_cand(c, best_candidate):
                best_candidate = c
        self.best_candidate = best_candidate