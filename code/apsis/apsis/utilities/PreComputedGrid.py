import logging
from apsis.models.ParamInformation import NominalParamDef, NumericParamDef
from apsis.models.Candidate import Candidate
import numpy as np
import pickle
import time
import math

class PreComputedGrid(object):
    grid_points = None
    param_defs = None
    dimensionality_per_param = None

    def __init__(self):
        self.grid_points = []

    def build_grid_points(self, param_defs, dimensionality):
        self.param_defs = param_defs
        dimensionality_per_param = None
        if not isinstance(dimensionality, list):
            dimensionality_per_param = [dimensionality]*len(param_defs)
        else:
            dimensionality_per_param = dimensionality

        num_grid_points = 1
        for i in range(len(dimensionality_per_param)):
            if isinstance(param_defs[i], NominalParamDef):
                dimensionality_per_param[i] = len(param_defs[i].values)

            num_grid_points *= dimensionality_per_param[i]
        self.dimensionality_per_param = dimensionality_per_param
        self.grid_points = self.grid_points_calculator(param_defs, dimensionality_per_param)
        for i in range(len(self.grid_points)):
            self.grid_points[i] = Candidate(self.grid_points[i])


    def grid_points_calculator(self, param_defs, dimensionalities_per_param):
        possible_values = []

        if isinstance(param_defs[0], NominalParamDef):
            possible_values = param_defs[0].values
        elif isinstance(param_defs[0], NumericParamDef):
            possible_values = np.linspace(0,1,dimensionalities_per_param[0]).tolist()

            for i in range(len(possible_values)):
                possible_values[i] = param_defs[0].warp_out(possible_values[i])

        #lowest level, one dimensional grid
        if len(param_defs) == 1 and len(dimensionalities_per_param) == 1:
            return [[x] for x in possible_values]

        #revursive level, link dimensions
        sub_grid = self.grid_points_calculator(param_defs[1:], dimensionalities_per_param[1:])

        return_grid = []

        for i in range(len(sub_grid)):
            for j in range(len(possible_values)):
                return_grid.append([possible_values[j]])
                return_grid[-1].extend(sub_grid[i])

        return return_grid

    def precompute_results(self, objective_func):
        for i in range(len(self.grid_points)):
            start_time = time.time()
            self.grid_points[i].result = objective_func(self.grid_points[i])
            end_time = time.time()
            duration = end_time - start_time
            self.grid_points[i].cost = duration

    def load_from_disk(self, filename):
        self.grid_points, self.param_defs = pickle.load(open(filename, "rb"))

    def save_to_disk(self, filename):
        pickle.dump((self.grid_points, self.param_defs), open(filename, "wb"))

    def get_closest_grid_candidate(self, candidate_in):
        closest_grid_candidate = self.grid_points[0]
        closest_distance = self.cand_distance(candidate_in, self.grid_points[0])

        possible_idx = self._get_possible_candidate_indices(candidate_in.params, 0)
        logging.debug("Possible IDs:" + str(possible_idx))
        for i in possible_idx:
            dist = self.cand_distance(candidate_in, self.grid_points[i])
            if dist < closest_distance:
                closest_grid_candidate = self.grid_points[i]
                closest_distance = dist
        #we found the closest grid candidate.
        return closest_grid_candidate

    def _get_possible_candidate_indices(self, param_vals, idx):
        possible_idx = []

        if isinstance(self.param_defs[idx], NominalParamDef):
            possible_idx = [self.param_defs.values.index(param_vals[0])]
        elif isinstance(self.param_defs[idx], NumericParamDef):
            val_warped = self.param_defs[idx].warp_in(param_vals[0])
            lower_bound = int(val_warped * (self.dimensionality_per_param[idx]-1))
            upper_bound = lower_bound + 1
            if upper_bound >= self.dimensionality_per_param[idx]:
                possible_idx = [lower_bound]
            else:
                possible_idx = [lower_bound, upper_bound]
            logging.debug("orig: %s, warped: %s, dim %s, lowerBnd %s" %(param_vals[0], val_warped, self.dimensionality_per_param[idx], lower_bound))

        #lowest level, one dimensional grid
        if len(param_vals) == 1:
            return possible_idx

        #revursive level, link dimensions
        sub_grid = self._get_possible_candidate_indices(param_vals[1:], idx+1)

        return_idx = []
        logging.debug("Adding idx %s to idx %s" %(possible_idx, sub_grid))
        for i in range(len(sub_grid)):
            for j in range(len(possible_idx)):

                cur_value = possible_idx[j] + sub_grid[i] * self.dimensionality_per_param[idx]
                return_idx.append(cur_value)

        return return_idx



    def evaluate_candidate(self, candidate_in):
        """
        ATTENTION: This method will change the Candidate object given in
        candidate_in!!

        Evaluate any given Candidate object against the nearest Candidate
        stored in this grid. The closest grid point to candidate_in is
        determined, then the param vector in candidate_in is updated and the
        result is added to candidate_in. This all happends in the very
        same object!

        Parameters
        ----------
        candidate_in: Candidate
            The candidate object that shall be evaluated using the pre-computed
            values in this PreComputedGrid.

            Attention: Again, this method will change the Candidate object
            given in candidate_in!!

        Returns
        -------
        candidate_in: Candidate
            The same object as given, but with possibly changed param vector.
        """
        closest_grid_candidate = self.get_closest_grid_candidate(candidate_in)
        candidate_in.result = closest_grid_candidate.result
        for i in range(len(candidate_in.params)):
            candidate_in.params[i] = closest_grid_candidate.params[i]
        logging.debug("In grid: " + str(closest_grid_candidate.result))
        logging.debug("In grid: " + str(closest_grid_candidate))

        return candidate_in

    def cand_distance(self, candidateA, candidateB):
        distance = 0
        for i in range(len(self.param_defs)):
            distance += (self.param_defs[i].distance(candidateA.params[i],
                                                    candidateB.params[i]))**2
        distance **= 0.5
        return distance