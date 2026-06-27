import triton
import triton.language as tl
import torch
import time

# matmul kernel without tiling
@triton.jit
def naive_matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K):
    # 1. Identify which output element this program computes
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 2. Compute coordinates of the output element
    offs_m = pid_m
    offs_n = pid_n

    # 3. Compute pointers to the input elements
    # This arithmetic assumes contiguous, row-major memory layout
    a_ptrs = a_ptr + offs_m * K
    b_ptrs = b_ptr + offs_n

    # 4. Create accumulator
    # Use a standard Python float for a scalar accumulator instead of a 1D tensor
    accumulator = 0.0

    # 5. Loop over K dimension
    for k in range(K):
        # Load scalar values
        a = tl.load(a_ptrs + k)
        b = tl.load(b_ptrs + k * N)

        accumulator += a * b

    # 6. Store result
    c_ptrs = c_ptr + offs_m * N + offs_n
    tl.store(c_ptrs, accumulator)

def naive_matmul(a, b):
    # Enforce contiguous memory layout. Without this, the manual pointer 
    # arithmetic in the kernel (e.g., `+ k * N`) will read the wrong memory 
    # addresses if the tensor was sliced or transposed.
    a = a.contiguous()
    b = b.contiguous()
    
    M, K = a.shape
    _, N = b.shape

    start = time.perf_counter()
    # Allocate output tensor
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    end = time.perf_counter()

    # Launch kernel using a 2D grid where every thread computes exactly 1 element
    grid = (M, N)
    naive_matmul_kernel[grid](a, b, c, M, N, K)

    return c, (end - start)

if __name__ == "__main__":
    # Define inputs
    a = [[1.0, 2.0], [3.0, 4.0]]
    b = [[5.0, 6.0], [7.0, 8.0]]
    
    a_tensor = torch.tensor(a, dtype=torch.float32, device='cuda')
    b_tensor = torch.tensor(b, dtype=torch.float32, device='cuda')
    
    #warmup runs
    for _ in range(10):
        naive_matmul(a_tensor, b_tensor)

    # Run Triton kernel
    c, exec_time = naive_matmul(a_tensor, b_tensor)
    print(f"Triton Naive MatMul Time: {exec_time:.6f} seconds")
    
    #torch warmup runs
    for _ in range(10):
        c_ref = a_tensor @ b_tensor
    #torch benchmark run
    start = time.perf_counter()
    c_ref = a_tensor @ b_tensor
    end = time.perf_counter()
    print(f"PyTorch MatMul Time: {end - start:.6f} seconds")

    # Run PyTorch reference
    c_ref = a_tensor @ b_tensor
    
    print("Triton Naive Result:")
    print(c)
    print("\nPyTorch Reference Result:")
    print(c_ref)

    # benchmarks for bigger matrices
    M, K, N = 20000, 20000, 20000
    a_tensor = torch.randn((M, K), dtype=torch.float32, device='cuda')
    b_tensor = torch.randn((K, N), dtype=torch.float32, device='cuda')

    #warmup runs
    for _ in range(10):
        naive_matmul(a_tensor, b_tensor)
    # Run Triton kernel
    c, exec_time = naive_matmul(a_tensor, b_tensor)
    print(f"Triton Naive MatMul Time: {exec_time:.6f} seconds")

    #torch warmup runs
    for _ in range(10):
        c_ref = a_tensor @ b_tensor
    #torch benchmark run
    start = time.perf_counter()
    c_ref = a_tensor @ b_tensor
    end = time.perf_counter()
    print(f"PyTorch MatMul Time: {end - start:.6f} seconds")

    #print the dimentions and size of the output matrices
    print(f"Triton Naive Result Shape: {c.shape}, Size: {c.numel()}")
    print(f"PyTorch Reference Result Shape: {c_ref.shape}, Size: {c_ref.numel()}")

    # there size on disk in mb
    print(f"Triton Naive Result Size on Disk: {c.element_size() * c.numel() / (1024 * 1024):.2f} MB")
    print(f"PyTorch Reference Result Size on Disk: {c_ref.element_size() * c_ref.numel() / (1024 * 1024):.2f} MB")