import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnums=(0,))
def kernel(TSTEPS: int, A: jax.Array, B: jax.Array):

    for t in range(1, TSTEPS):
        B = B.at[1:-1, 1:-1].set(0.2 * (A[1:-1, 1:-1] + A[1:-1, :-2] + A[1:-1, 2:] +
                               A[2:, 1:-1] + A[:-2, 1:-1]))
        A = A.at[1:-1, 1:-1].set(0.2 * (B[1:-1, 1:-1] + B[1:-1, :-2] + B[1:-1, 2:] +
                               B[2:, 1:-1] + B[:-2, 1:-1]))
    
    return A, B
