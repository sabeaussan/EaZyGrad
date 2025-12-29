import numpy as np
import time
import numba as nb

# Dilation using Kronecker product
def dilate_array_kron(array, dilation=2):
    if dilation > 1:
        size = array.shape[-1]
        eff_size = size + (size-1)*(dilation-1)
        dilation_matrix = np.zeros((dilation, dilation), dtype=np.float32)
        dilation_matrix[0,0]=1
        return np.ascontiguousarray(np.kron(array, dilation_matrix)[..., :eff_size, :eff_size])
    else:
        return array

@nb.njit(fastmath=True, parallel=True)
def dilate_array_jit(array, dilation_factor=2):
    B, C, H, W = array.shape
    dilated_spatial_size = H * dilation_factor - (dilation_factor - 1)
    
    result = np.zeros((B, C, dilated_spatial_size, dilated_spatial_size), dtype=np.float32)
    
    for b in nb.prange(B):  # Parallelize over the batch dimension
        for c in nb.prange(C):  # Parallelize over the channel dimension
            for i in range(H):  # Parallelize over the height
                for j in range(W):  # Parallelize over the width
                    result[b, c, i * dilation_factor, j * dilation_factor] = array[b, c, i, j]
    
    return result

# Benchmarking function
def benchmark_dilation(array, dilation_factor=2, n_runs=10):
    # Warm up JIT compilation for Numba function
    dilate_array_jit(array, dilation_factor)

    # Benchmark dilate_array_kron
    start_time = time.time()
    for _ in range(n_runs):
        _ = dilate_array_kron(array, dilation_factor)
    kron_time = time.time() - start_time

    # Benchmark dilate_array_jit
    start_time = time.time()
    for _ in range(n_runs):
        _ = dilate_array_jit(array, dilation_factor)
    jit_time = time.time() - start_time

    return kron_time, jit_time

# Example usage
array = np.random.rand(150, 16, 66, 34).astype(np.float32)  # Example of a 4D array
dilation_factor = 2
n_runs = 10

kron_time, jit_time = benchmark_dilation(array, dilation_factor, n_runs)

print(f"Kronecker dilation average time over {n_runs} runs: {kron_time / n_runs:.6f} seconds per run")
print(f"JIT dilation average time over {n_runs} runs: {jit_time / n_runs:.6f} seconds per run")
