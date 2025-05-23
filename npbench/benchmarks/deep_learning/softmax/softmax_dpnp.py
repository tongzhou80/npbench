import dpnp as np

# Numerically-stable version of softmax using DPNP
def softmax(x):
    tmp_max = np.max(x, axis=-1, keepdims=True)
    tmp_out = np.exp(np.subtract(x, tmp_max)) # Optimized subtraction!
    tmp_sum = np.sum(tmp_out, axis=-1, keepdims=True)
    return tmp_out / tmp_sum
