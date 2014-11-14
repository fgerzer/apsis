from apsis.models.ParamInformation import NominalParamDef, NumericParamDef
from apsis.models.Candidate import Candidate
import numpy as np
import pickle

class PreComputedGrid(object):
    grid_points = None

    def __init__(self):
        self.grid_points = []

    def build_grid_points(self, param_defs, dimensionality):
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
            self.grid_points[i].result = objective_func(self.grid_points[i])

    def load_from_disk(self, filename):
        self.grid_points = pickle.load(open(filename, "rb"))

    def save_to_disk(self, filename):
        pickle.dump(self.grid_points, open(filename, "wb"))



