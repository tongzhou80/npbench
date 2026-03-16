import numpy as np
import appy


@appy.jit
def kernel(alpha, beta, A, u1, u2, v1, v2, w, x, y, z):
    M, N = A.shape
    #pragma parallel for
    for i in range(M):
        A[i, :] += u1[i] * v1[:] + u2[i] * v2[:]

    #pragma parallel for
    for j in range(N):
        x[j] += beta * (y[:] @ A[:, j]) + z[j]

    #pragma parallel for
    for i in range(M):
        w[i] += alpha * (A[i, :] @ x[:])

    return A, x, w
