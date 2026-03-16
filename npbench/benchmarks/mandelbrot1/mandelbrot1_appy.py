import numpy as np
import appy


@appy.jit
def mandelbrot(xmin, xmax, ymin, ymax, xn, yn, maxiter, horizon=2.0):
    X = np.linspace(xmin, xmax, xn, dtype=np.float32)
    Y = np.linspace(ymin, ymax, yn, dtype=np.float32)
    N = np.zeros(yn * xn, dtype=np.int32)

    #pragma parallel for
    for idx in range(yn * xn):
        i = idx // xn
        j = idx % xn
        cr = X[j]
        ci = Y[i]
        zr = 0.0
        zi = 0.0
        n = 0
        for k in range(maxiter):
            if zr * zr + zi * zi >= horizon * horizon:
                break
            new_zr = zr * zr - zi * zi + cr
            new_zi = 2.0 * zr * zi + ci
            zr = new_zr
            zi = new_zi
            n = k
        N[idx] = n

    N[N == maxiter - 1] = 0
    return N.reshape(yn, xn)
