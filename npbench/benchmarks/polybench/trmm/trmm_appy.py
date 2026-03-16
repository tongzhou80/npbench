import numpy as np
import appy


@appy.jit
def kernel(alpha, A, B):
    M, N = B.shape
    for i in range(M):
        #pragma parallel for
        for j in range(N):
            B[i, j] += A[i + 1:, i] @ B[i + 1:, j]
    B *= alpha
    return B
