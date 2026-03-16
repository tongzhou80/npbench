import numpy as np
import appy


@appy.jit
def kernel(path):
    for k in range(path.shape[0]):
        #pragma parallel for
        for i in range(path.shape[0]):
            path[i, :] = np.minimum(path[i, :], path[i, k] + path[k, :])
    return path
