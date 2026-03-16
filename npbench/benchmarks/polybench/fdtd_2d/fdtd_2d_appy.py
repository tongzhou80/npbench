import numpy as np
import appy


@appy.jit
def kernel(TMAX, ex, ey, hz, _fict_):
    for t in range(TMAX):
        ey[0, :] = _fict_[t]
        #pragma parallel for
        for i in range(1, ey.shape[0]):
            #pragma simd
            for j in range(ey.shape[1]):
                ey[i, j] -= 0.5 * (hz[i, j] - hz[i - 1, j])
        #pragma parallel for
        for i in range(ex.shape[0]):
            #pragma simd
            for j in range(1, ex.shape[1]):
                ex[i, j] -= 0.5 * (hz[i, j] - hz[i, j - 1])
        #pragma parallel for
        for i in range(hz.shape[0] - 1):
            #pragma simd
            for j in range(hz.shape[1] - 1):
                hz[i, j] -= 0.7 * (ex[i, j + 1] - ex[i, j] +
                                    ey[i + 1, j] - ey[i, j])
    return ey, ex, hz
