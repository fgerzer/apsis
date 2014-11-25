import logging
from apsis.models.ParamInformation import NominalParamDef, NumericParamDef
from apsis.models.Candidate import Candidate
import numpy as np
import pickle
import time
import math


class PreComputedGrid(object):
    """
    The PreComputedGrid class represents a precomputed grid of a
    multidimensional function. It stores - using the Candidate object - results
    and costs for each of the points. Additionally, it supports loading and
    storing its grid.
    This class can be used for precomputing expensive functions and quickly
    evaluating optimizers on there.

    Attributes
    ----------
    grid_points: list of Candidate
        This is a list of candidate objects representing grid points that have
        been evaluated. Since each candidate stores parameters, results and
        costs, it is very suitable for this.
    param_defs: list of ParamDef
        A list of the parameter definitions for this grid.
    dimensionality_per_param: list of int
        represents how many samples per parameter should be used. Note that
        the number of grid_points increases with
        dimensionality_per_param**params.

    Notes
    -----
    To use a grid, several functions have to be called in a specific order.
    To make a new grid, you first have to assign the object using the __init__
     function. Afterwards, build_grid_points initializes the grid points
     themselves, but does not yet calculate the results. precompute_results
     does this. It is also recommended to save the grid using save_to_disk.
    In contrast, loading a grid is significantly easier. After assigning the
     object, load_from_disk loads the complete precomputed grid.
    """

    grid_points = None
    param_defs = None
    dimensionality_per_param = None

    def __init__(self):
        """
        Initializes a PrecomputedGrid object. Its grid is empty.
        """
        self.grid_points = []

    def build_grid_points(self, param_defs, num_pts):
        """
        Constructs the grid points the PreComputedGrid consists of.

        Parameters
        ----------
        param_defs: list of ParamDefs
            Defines the parameters used in the grid. The order is used to
             determine the grid order.
        num_pts: int or list of int
            How many points per grid_dimension there are.
            If specified as an integer, all dimensions have the same number of
            points. Otherwise, these numbers may vary.
            Note that this number is irrelevant for categorial data. In this
            case, the parameter is just iterating through all possible values.

        Notes
        -----
        Note that the number of grid points and therefore both memory needs and
         computational complexity grow in the order of
         dimensionality**number of parameters.
        """
        self.param_defs = param_defs
        if not isinstance(num_pts, list):
            num_pts_per_param = [num_pts] * len(param_defs)
        else:
            num_pts_per_param = num_pts

        num_grid_points = 1
        for i in range(len(num_pts_per_param)):
            # This corrects the number of values for nominal definitions.
            if isinstance(param_defs[i], NominalParamDef):
                num_pts_per_param[i] = len(param_defs[i].values)

            num_grid_points *= num_pts_per_param[i]
        self.dimensionality_per_param = num_pts_per_param
        self.grid_points = self.grid_points_calculator(param_defs,
                                                       num_pts_per_param)
        for i in range(len(self.grid_points)):
            self.grid_points[i] = Candidate(self.grid_points[i])


    def grid_points_calculator(self, param_defs, num_pts):
        """
        Calculates - recursively - grid_points.

        To do so, the function calculates all possible values for the first
         entry of param_defs, then recursively calculates a grid for all the
         other entries, followed by a cross-product of the two.
        Parameters
        ----------
        param_defs: list of ParamDef
            The param_defs for which to calculate the grid.
        num_pts: list of int
            The number of points each dimension can take. Has to have the same
             dimension as param_defs.

        Returns
        -------
        return_grid: list of values
            A list of all the values that the grid points can take.
        """
        values = []

        if isinstance(param_defs[0], NominalParamDef):
            values = param_defs[0].values
        elif isinstance(param_defs[0], NumericParamDef):
            values = np.linspace(0, 1, num_pts[0]).tolist()

            for i in range(len(values)):
                values[i] = param_defs[0].warp_out(values[i])

        # lowest level, one dimensional grid
        if len(param_defs) == 1 and len(num_pts) == 1:
            return [[x] for x in values]

        #recursive level,
        sub_grid = self.grid_points_calculator(param_defs[1:], num_pts[1:])

        return_grid = []
        #And form the cross-product.
        for i in range(len(sub_grid)):
            for j in range(len(values)):
                return_grid.append([values[j]])
                return_grid[-1].extend(sub_grid[i])

        return return_grid

    def precompute_results(self, objective_func, obj_func_args=None):
        """
        This precomputes the result for every entry of grid_points.

        In general, it calls each function using the obj_func_args and the
         grid_points entry. It assumes that the return of objective_function is
         a float value. It also times the execution.

        Parameters
        ----------
        objective_func: function
            This function represents the function to evaluate. Its first
             parameter has to be a candidate object (which should not be
             changed), and the other parameters have to remain the same from
             evaluation to evaluation.
             It has to return a float value representing the quality of the
             result.
        obj_func_args: dict of function arguments
            Optional function arguments for objective_func.
        """
        if obj_func_args is None:
            obj_func_args = {}
        for i in range(len(self.grid_points)):
            start_time = time.time()
            self.grid_points[i].result = objective_func(self.grid_points[i],
                                                        **obj_func_args)
            end_time = time.time()
            duration = end_time - start_time
            self.grid_points[i].cost = duration

    def load_from_disk(self, filename):
        """
        Loads a pickled grid from disk.

        Parameters
        ----------
        filename: string
            The string representation of the file to read from.
        """
        self.grid_points, self.param_defs, self.dimensionality_per_param = \
            pickle.load(open(filename, "rb"))

    def save_to_disk(self, filename):
        """
        Stores a pickled grid to the disk.

        Parameters
        ----------
        filename: string
            The string representation of the file to save to.
        """
        pickle.dump(
            (self.grid_points, self.param_defs, self.dimensionality_per_param),
            open(filename, "wb"))

    def get_closest_grid_candidate(self, candidate_in):
        """
        For candidate_in, find the most similar precomputed point.

        The goal behind this is to avoid computing an entirely new point,
         thereby allowing a much faster comparison.

        Parameters
        ----------
        candidate_in: Candidate
            The candidate to find a similar one to.

        Returns
        -------
        closest_grid_candidate: Candidate
            The most similar candidate in the grid.
        """
        closest_grid_candidate = self.grid_points[0]
        closest_distance = float("inf")

        possible_idx = self._get_close_candidate_indices(
            candidate_in.params)
        logging.debug("Possible IDs:" + str(possible_idx))
        for i in possible_idx:
            dist = self.cand_distance(candidate_in, self.grid_points[i])
            if dist < closest_distance:
                closest_grid_candidate = self.grid_points[i]
                closest_distance = dist
        logging.debug(
            "Possible candidate params: %s" % str(closest_grid_candidate))
        return closest_grid_candidate

    def _get_close_candidate_indices(self, param_vals, idx=0):
        """
        This recursively gives all possible indices for the closest grid
        candidate.

        This can be easily done because they are ordered by the values of the
         parameters in their order.

        Parameters
        ----------
        param_vals: list of values
            The values the parameters take.

        idx: int
            Keeps the current position in the list. To start the recursion,
            set it to 0 (which is also the default value)

        Returns
        -------
        return_idxs: list of ints
            The possible indices param_vals can be close to.
        """
        possible_idx = []

        if isinstance(self.param_defs[idx], NominalParamDef):
            possible_idx = [self.param_defs.values.index(param_vals[0])]
        elif isinstance(self.param_defs[idx], NumericParamDef):
            val_warped = self.param_defs[idx].warp_in(param_vals[0])
            lower_bound = int(
                val_warped * (self.dimensionality_per_param[idx] - 1))
            upper_bound = lower_bound + 1
            if upper_bound >= self.dimensionality_per_param[idx]:
                possible_idx = [lower_bound]
            else:
                possible_idx = [lower_bound, upper_bound]
            logging.debug("orig: %s, warped: %s, dim %s, lowerBnd %s" % (
                param_vals[0], val_warped, self.dimensionality_per_param[idx],
                lower_bound))

        # lowest level, one dimensional grid
        if len(param_vals) == 1:
            return possible_idx

        #revursive level
        sub_grid = self._get_close_candidate_indices(param_vals[1:], idx + 1)

        return_idx = []
        logging.debug("Adding idx %s to idx %s" % (possible_idx, sub_grid))
        for i in range(len(sub_grid)):
            for j in range(len(possible_idx)):
                cur_value = possible_idx[j] + sub_grid[i] * \
                                              self.dimensionality_per_param[
                                                  idx]
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
        """
        Gives the distance between the parameters of candidateA and candidateB.

        This is defined as the Euclidian Distances between their parameters.
        Parameters
        ----------
        candidateA: Candidate
            The first candidate
        candidateB: Candidate
            The second candidate.

        Returns
        -------
        distance: float
            The euclidian distance between the candidates' parameters
        """
        distance = 0
        for i in range(len(self.param_defs)):
            distance += (self.param_defs[i].distance(candidateA.params[i],
                                                     candidateB.params[
                                                         i])) ** 2
        distance **= 0.5
        return distance