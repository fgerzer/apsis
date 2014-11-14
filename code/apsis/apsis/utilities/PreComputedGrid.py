import logging
from apsis.models.ParamInformation import NominalParamDef, NumericParamDef
from apsis.models.Candidate import Candidate
import numpy as np
import pickle
import time

class PreComputedGrid(object):
    grid_points = None
    param_defs = None

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
                return_grid.append([possible_values[j]].extend(sub_grid[i]))

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
        for i in range(len(self.grid_points)):
            dist = self.cand_distance(candidate_in, self.grid_points[i])
            if dist < closest_distance:
                closest_grid_candidate = self.grid_points[i]
                closest_distance = dist
        #we found the closest grid candidate.
        return closest_grid_candidate

    def evaluate_candidate(self, candidate_in):
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