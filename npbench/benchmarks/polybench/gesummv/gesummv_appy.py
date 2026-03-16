import numpy as np
import appy


@appy.jit
def kernel(alpha, beta, A, B, x):
    y = np.empty((A.shape[0],), dtype=x.dtype)
    #pragma parallel for
    for i in range(A.shape[0]):
        y[i] = np.sum(alpha * A[i, :] * x[:] + beta * B[i, :] * x[:])
    return y
