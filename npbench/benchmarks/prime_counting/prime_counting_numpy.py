import numpy as np


def prime_count(N):
    """Count primes up to N using trial division."""
    is_prime = np.zeros(N, dtype=np.int32)
    for n in range(2, N):
        prime = True
        for d in range(2, int(n ** 0.5) + 1):
            if n % d == 0:
                prime = False
                break
        if prime:
            is_prime[n] = 1
    return np.sum(is_prime)
