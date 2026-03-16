import numpy as np
import appy


@appy.jit
def kernel(A):
    M, N = A.shape
    Q = np.zeros((M, N), dtype=A.dtype)
    R = np.zeros((N, N), dtype=A.dtype)
    for k in range(N):
        nrm = A[:, k] @ A[:, k]
        R[k, k] = np.sqrt(nrm)
        Q[:, k] = A[:, k] / R[k, k]
        #pragma parallel for
        for j in range(k + 1, N):
            s = Q[:, k] @ A[:, j]
            R[k, j] = s
            A[:, j] -= Q[:, k] * s
    return Q, R
