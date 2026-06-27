import triton
import triton.language as tl
import torch
import time

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # 1. Identify which output tile this program computes
    pid = tl.program_id(0)

    # calculating the cordinates
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    # logic for this is:
    # pid_m = pid // num_pid_n, as we are dividing the total number of program ids by the number of program ids in the n dimension to get the program id in the m dimension
    # pid_n = pid % num_pid_n, as we are taking the modulus of the total number of program ids by the number of program ids in the n dimension to get the program id in the n dimension
    """
    lets visualize the thing above
    suppose we have a matrix of size MxN, and we want to compute the matrix multiplication of A and B, where A is of size MxK and B is of size KxN. We can divide the output matrix C into tiles of size BLOCK_M x BLOCK_N. The number of tiles in the n dimension is num_pid_n = ceil(N / BLOCK_N). The program id pid is a linear index that ranges from 0 to (num_pid_m * num_pid_n - 1), where num_pid_m = ceil(M / BLOCK_M). We can compute the program id in the m dimension as pid_m = pid // num_pid_n, and the program id in the n dimension as pid_n = pid % num_pid_n. This way, we can map each program id to a specific tile in the output matrix C.
    lets se it on a example of 4x4 matrix with BLOCK_M=2 and BLOCK_N=2
    suppose we have a matrix of size 4x4, and we want to compute the matrix multiplication of A and B, where A is of size 4x4 and B is of size 4x4. We can divide the output matrix C into tiles of size 4x4. The number of tiles in the n dimension is num_pid_n = ceil(4 / 2) = 2. The program id pid is a linear index that ranges from 0 to (num_pid_m * num_pid_n - 1) = (2 * 2 - 1) = 3, where num_pid_m = ceil(4 / 2) = 2. We can compute the program id in the m dimension as pid_m = pid // num_pid_n, and the program id in the n dimension as pid_n = pid % num_pid_n. This way, we can map each program id to a specific tile in the output matrix C.   
    """

    # 2. Compute tile coordinates
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # 3. Compute pointers
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    # B is (K x N), its blocks move vertically down K
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # 4. Create accumulator
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # 5. Loop over K dimension
    for k in range(0, K, BLOCK_K):

        # Boundaries check: mask out elements if matrix dimensions are not multiples of block sizes
        a_mask = (offs_m[:, None] < M) & ((k + offs_k[None, :]) < K)
        b_mask = ((k + offs_k[:, None]) < K) & (offs_n[None, :] < N)

        # Load blocks into SRAM
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Perform matrix multiply-accumulate
        accumulator += tl.dot(a, b)

        # Advance pointers along the K dimension for the next iteration
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # 6. Store result
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Ensure inputs are contiguous and on the GPU
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA"
    assert a.shape[1] == b.shape[0], "Incompatible dimensions for matrix multiplication"

    M, K = a.shape
    _, N = b.shape

    # Allocate output tensor
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Define execution block sizes (Tuned hyper-parameters)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    # 1D Grid mapping: Total output blocks needed
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    start = time.perf_counter()
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    end = time.perf_counter()
    return c, (end - start)

# --- Validation Driver ---
if __name__ == "__main__":
    torch.manual_seed(42)
    M, K, N = 20000, 20000, 20000  # Large matrix sizes for benchmarking
    
    # Generate random FP32 matrices on GPU
    A = torch.randn((M, K), device="cuda", dtype=torch.float32)
    B = torch.randn((K, N), device="cuda", dtype=torch.float32)
    
    # triton warmup runs
    for _ in range(10):
        matmul(A, B)
    # benchmarking runs
    c, triton_time = matmul(A, B)
    print(f"Triton MatMul Time: {triton_time:.6f} seconds")

    # torch warmup runs
    for _ in range(10):
        C_torch = torch.matmul(A, B)
    # torch benchmark run
    start = time.perf_counter()
    C_torch = torch.matmul(A, B)
    end = time.perf_counter()
    print(f"PyTorch MatMul Time: {end - start:.6f} seconds")

    # Run Triton Matmul
    C_triton, _ = matmul(A, B)
    
    # Run PyTorch Native Matmul
    C_torch = torch.matmul(A, B)
    
    # Check correctness
    if torch.allclose(C_triton, C_torch, atol=1e-3, rtol=1e-3):
        print("Success! Triton results match PyTorch closely.")
    else:
        print("Deviation detected between Triton and PyTorch results.")