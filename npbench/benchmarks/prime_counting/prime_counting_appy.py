import numpy as np
import appy


@appy.jit
def prime_count(N):
    is_prime = np.zeros(N, dtype=np.int32)

    #pragma parallel for
    for n in range(2, N):
        prime = 1
        for d in range(2, n):
            if d * d > n:
                break
            if n % d == 0:
                prime = 0
                break
        is_prime[n] = prime

    return np.sum(is_prime)
