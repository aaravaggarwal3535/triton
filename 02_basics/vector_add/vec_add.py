import torch
import triton
import triton.language as tl
import time

DEVICE = triton.runtime.driver.active.get_active_torch_device() # get the active cuda device

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    This is kernal to add 2 vectors(tensors) of size n elements. 
    x_ptr: pointer to first vector
    y_ptr: pointer to second vector
    output_ptr: pointer to output vector
    n_elements: number of elements in the vector
    BLOCK_SIZE: size of each block(means the kernel will process the vector of 1024 elements at the same time), tl.constexpr means that this value is known at compile time and cannot be changed at runtime.
    """
    pid = tl.program_id(axis = 0)
    offsets = (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE))

    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask = mask)
    y = tl.load(y_ptr + offsets, mask = mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask = mask)

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
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output

torch.manual_seed(0)
size = 10000000
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