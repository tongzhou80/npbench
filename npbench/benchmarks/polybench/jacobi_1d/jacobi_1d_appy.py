import numpy as np
import appy


@appy.jit
def kernel(TSTEPS, A, B):
    for t in range(1, TSTEPS):
        #pragma parallel for
        for i in range(1, A.shape[0] - 1):
            B[i] = 0.33333 * (A[i - 1] + A[i] + A[i + 1])
        #pragma parallel for
        for i in range(1, B.shape[0] - 1):
            A[i] = 0.33333 * (B[i - 1] + B[i] + B[i + 1])
    return A, B
