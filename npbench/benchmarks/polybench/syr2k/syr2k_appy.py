import numpy as np
import appy


@appy.jit
def kernel(alpha, beta, C, A, B):
    #pragma parallel for
    for i in range(A.shape[0]):
        C[i, :i + 1] *= beta
        for k in range(A.shape[1]):
            C[i, :i + 1] += (A[i, k] * alpha * B[:i + 1, k] +
                              B[i, k] * alpha * A[:i + 1, k])
    return C
