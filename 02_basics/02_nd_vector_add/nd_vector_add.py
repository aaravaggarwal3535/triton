# triton kernel to add nd matrix
import torch
import time
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device() # get the active cuda device

@triton.jit
def kernel_nd_vector_add(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    # program id
    pid = tl.program_id(0)
    # offsets for each thread in the block
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # mask to avoid out-of-bounds memory access
    mask = offsets < n_elements
    # load x and y values from memory
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    # compute the sum
    output = x + y
    # store the result back to memory
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    This function is used to add 2 vectors(tensors) of size n elements. 
    x: first vector
    y: second vector
    return: output vector
    """
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    kernel_nd_vector_add[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output

torch.manual_seed(0)
size = (1024, 1024)
x = torch.randn(size, device=DEVICE)
y = torch.randn(size, device=DEVICE)

# 1. WARMUP & COMPILATION (Discard these runs)
_ = add(x, y) 
_ = x + y
_ = x.cpu() + y.cpu()
torch.cuda.synchronize() # Wait for everything to clear out

torch.cuda.synchronize() # Wait for everything to clear out
start = time.perf_counter()
output_triton = add(x, y)
torch.cuda.synchronize() # Wait for everything to clear out
end = time.perf_counter()
print(f'Triton time: {(end - start) * 1000:.4f} in milli seconds')

start = time.perf_counter()
output_torch = x + y
torch.cuda.synchronize() # Wait for everything to clear out
end = time.perf_counter()
print(f'Torch time: {(end - start) * 1000:.4f} in milli seconds')

#cpu addition with time
start = time.perf_counter()
output_cpu = x.cpu() + y.cpu()
torch.cuda.synchronize() # Wait for everything to clear out
end = time.perf_counter()
print(f'CPU time: {(end - start) * 1000:.4f} in milli seconds')

print(output_torch == output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')