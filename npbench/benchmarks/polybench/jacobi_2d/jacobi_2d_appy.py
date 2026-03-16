import numpy as np
import appy


@appy.jit
def kernel(TSTEPS, A, B):
    for t in range(1, TSTEPS):
        #pragma parallel for
        for i in range(1, A.shape[0] - 1):
            #pragma simd
            for j in range(1, A.shape[1] - 1):
                B[i, j] = 0.2 * (A[i, j] + A[i, j - 1] + A[i, j + 1] +
                                  A[i - 1, j] + A[i + 1, j])
        #pragma parallel for
        for i in range(1, B.shape[0] - 1):
            #pragma simd
            for j in range(1, B.shape[1] - 1):
                A[i, j] = 0.2 * (B[i, j] + B[i, j - 1] + B[i, j + 1] +
                                  B[i - 1, j] + B[i + 1, j])
    return A, B
