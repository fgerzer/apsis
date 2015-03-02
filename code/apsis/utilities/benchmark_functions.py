import math
from apsis.utilities.randomization import check_random_state
from scipy.stats.distributions import norm
import numpy as np

def branin_func(x, y, a=1, b=5.1/(4*math.pi**2), c=5/math.pi, r=6, s=10,
                t=1/(8*math.pi)):
        """
        Branin hoo function.

        This is the same function as in
        http://www.sfu.ca/~ssurjano/branin.html. The default parameters are
        taken from that same site.

        With the default parameters, there are three minima with f(x)=0.397887:
        (-pi, 12.275), (pi, 2.275) and (9.42478, 2.475).

        Parameters
        ---------
        x : float
            A real valued float
        y : float
            A real valued float
        a, b, c, r, s, t : floats, optional
            Parameters for the shape of the Branin hoo function. Thier default
            values are according to the recommendations of the above website.
        Returns
        -------
        result : float
            A real valued float.
        """
        result = a*(y-b*x**2+c*x-r)**2 + s*(1-t)*math.cos(x)+s
        return result


def gen_one_dim_noise(y_min=0, y_max=1, points=100, random_state=None):
    """
    Generates a noise_gen array as used in one_dim_noise.

    Parameters
    ----------
    y_min, y_max : float, optional
        The minimum and maximum values for the result. Standard is between 0
        and 1.
    points : int, optional
        The number of points used for interpolation.
    random_state : numpy randomstate or int, optional
        The random state to use. Used for reproducability.

    Returns
    -------
    noise_gen : list
        The generated points.
    """
    random_state = check_random_state(random_state)
    noise_gen = []
    for i in range(points):
        noise_gen.append(random_state.uniform(y_min, y_max))
    return noise_gen

def gen_min_max_one_dim_noise(noise_gen, variance=1):
    x_min = float("inf")
    x_max = float("-inf")
    for i in range(len(noise_gen)):
        x_val, _ = one_dim_noise(float(i)/len(noise_gen), variance=variance, noise_gen=noise_gen)
        if x_val < x_min:
            x_min = x_val
        if x_val > x_max:
            x_max = x_val
    return x_min, x_max

def one_dim_noise(x, variance=1, noise_gen=None, x_min=0, x_max=1):
    """
    Generates a one-dimensional noise.

    If noise_gen is None, a new noise_gen is generated.
    This uses a list of points (from noise_gen). Each point to be evaluated is
    then generated from all the surrounding points weighted by a gaussian
    distribution with zero mean and variance variance.

    Noise-gen is assumed to be a uniformly distributed list of points in the
    0-1 hypercube.

    Parameters
    ----------
    x : float
        The value on which to evaluate
    variance : float, optional
        The variance used to interpolate between the points.
    noise_gen : array, optional
        The points to be interpolated between.

    Returns
    -------
    result : float
        The value at point x
    noise_gen : list
        Either the parameter noise_gen or (when it is None) a new list.
    """
    if noise_gen is None:
        noise_gen = gen_one_dim_noise()

    x_value = 0
    prob_sum = 0
    step = 1./float(len(noise_gen))

    gaussian = norm(scale=variance)

    for i in range(len(noise_gen)):
        i_corrected = (i * step - x)/variance
        prob = gaussian.pdf(i_corrected)
        prob_sum += prob
        x_value += prob * noise_gen[i]
    x_value /= prob_sum
    x_value = (x_value - x_min)/(x_max- x_min)
    return x_value, noise_gen

def gen_two_dim_noise(y_min=0, y_max=1, points=100, random_state=None):
    """
    Generates a noise_gen array as used in one_dim_noise.

    Parameters
    ----------
    y_min, y_max : float, optional
        The minimum and maximum values for the result. Standard is between 0
        and 1.
    points : int, optional
        The number of points used for interpolation.
    random_state : numpy randomstate or int, optional
        The random state to use. Used for reproducability.

    Returns
    -------
    noise_gen : list
        The generated points.
    """
    random_state = check_random_state(random_state)
    noise_gen = np.zeros(points, points)
    for i in range(points):
        for j in range(points):
            noise_gen[i, j] = random_state.uniform(y_min, y_max)
    return noise_gen

def gen_noise(dims, points, random_state=None):
    """
    Generates an ndarray representing random noise.

    This array has dims dimensions and points points per dimension. Each
    element is between 0 and 1.

    Parameters
    ----------
    dims : int
        The dimensionality of the noise.
    points : int
        The number of points per dimension.
    random_state : numpy RandomState
        The random state to generate the noise.

    Returns
    -------
    noise_gen : ndarray
        The ndarray containing the noise.
    """
    random_state = check_random_state(random_state)
    dimension_tuple = (points,)*dims
    noise_gen = random_state.rand(*dimension_tuple)
    return noise_gen

def get_noise_value_at(x, variance, noise_gen, val_min=0, val_max=1):
    """
    Returns the noise value for noise_gen for a given variance at x.

    The noise_gen is assumed to represent a [0, 1] hypercube and is smoothed
    by a gaussian distribution with variance variance.

    Note that the smoothing is hard-capped at a 3 sigma interval due to
    performance reasons.

    Parameters
    ----------
    x : list of real values
        The values of x. The ith entry represents the value of x in the ith
        dimension.
    variance : float
        The variance of the normal distribution to smooth the noise.
    noise_gen : ndarray
        The array representing the generated noise.
    val_min, val_max : float
        This is used to scale the actual maximum and minimum values to represent
        the same as otherwise values would not be comparable between variances.

    Returns
    -------
    x_value : float
        The value of the function at the point x.
    """
    x_value = 0
    prob_sum = 0
    gaussian = norm(scale=variance)
    dims = len(noise_gen.shape)
    points = len(noise_gen[0])

    closest_idx = _gen_closest_index(x, points)
    close_indices = _gen_close_indices(closest_idx, max(1, int(variance*3*points)),
                                      dims, points)
    for i in close_indices:
        dist = _calc_distance_grid(x, i, points)
        prob = gaussian.pdf(dist)
        prob_sum += gaussian.pdf(dist)
        x_value += prob * noise_gen[i]
    x_value /= prob_sum

    x_value = (x_value - val_min)/(val_max- val_min)

    return x_value

def _calc_distance_grid(x_coords, y_indices, points):
    """
    Calculates the euclidian distance between two points for a certain grid.

    Parameters
    ----------
    x_coords, y_indices : list
        The points in a list format for which the distance should be
        calculated. Note that x_coords is in [0, 1] coords, while y_indices is
        in an index format, that is dependant on the number of points.
        The entries x and y are indices of the grid, so their
         final distance is dependant on the number of points.
    points : int
        The number of points per dimension on the grid.

    Returns
    -------
    distance : float
        The distance between x and y.
    """
    distance = 0
    for i in range(len(x_coords)):
        distance += (float(x_coords[i]) - float(y_indices[i])/points)**2
    return distance

def _gen_closest_index(x, points):
    """
    Generates the closes index to the point x.

    Not that this is not a hard index, but may vary up to +/- 1 in each dim.

    Parameters
    ----------
    x : list
        The [0, 1] hypercube coordinates for x.
    points : int
        The number of points in each dimension.

    Returns
    -------
    closest_index : tuple
        Indexing tuple for the closest point.
    """
    closest_index = []
    for i in range(len(x)):
        closest_index.append(int(x[i]*points))
    return tuple(closest_index)

def _gen_close_indices(x_indices, max_dist, dims, points):
    """
    Generates a list of closest indices to consider for the noise smoothing.

    Parameters
    ----------
    x_indices : list
        The list of indices for each dimension around which to consider the
        indices.
    max_dist : int
        The maximum distance (in indices) around x for which to consider items.
    dims : int
        The dimensions of x.
    points : int
        The number of points per dimension

    Returns
    -------
    list_indices : list of tuples
        A list of tuples as indices which are closest to x.
    """
    raw_list_indices = _gen_close_indices_rec(x_indices, max_dist, dims, points)
    list_indices = []
    for l in raw_list_indices:
        acceptable = True
        for d in range(dims):
            if 0 > l[d] or l[d] >= points:
                acceptable = False
                break
        if acceptable:
            list_indices.append(tuple(l))
    return list_indices

def _gen_close_indices_rec(x, max_dist, dims, points):
    """
    Recursively generates a list of closest indices to consider for the noise smoothing.

    Parameters
    ----------
    x_indices : list
        The list of indices for each dimension around which to consider the
        indices.
    max_dist : int
        The maximum distance (in indices) around x for which to consider items.
    dims : int
        The dimensions of x.
    points : int
        The number of points per dimension

    Returns
    -------
    list_indices : list of tuples
        A list of lists as indices which are closest to x.
    """
    list_indices = []
    if len(x) == 1:
        for i in range(-max_dist, max_dist+1):
            list_indices.append([int(i + x[0])])
    else:
        list_prev_dim = _gen_close_indices_rec(x[1:], max_dist, dims-1, points)
        for i in range(len(list_prev_dim)):
            for j in range(-max_dist, max_dist+1):
                to_append = [int(j + x[0])] + list_prev_dim[i][:]
                #to_append = [int(j + x[0])] + list_prev_dim[i]
                list_indices.append(to_append)
    return list_indices
