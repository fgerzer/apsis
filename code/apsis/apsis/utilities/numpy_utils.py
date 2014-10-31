import numpy as np

def ensure_np_array(input):
    #a bit hacky, but for 1d check if array, if not make one manually
    if not isinstance(input, np.ndarray):
        input = np.asarray([input])

    return input