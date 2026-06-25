# Vector Add

## Imports

### triton
```
Used for launching kernels
Autotuning
JIT compilation
```

### triton.language (tl)
```
GPU operations
load/store
arithmetic
program ids
reductions
```

## @triton.jit
```
used to compile the code on gpu instead on cpu
```

## tl.program_id()\
```
this is basically means 
if N is 1024
block size is 256
then there will be 4 program so program id 1, 2, 3, 4
```
```c
// code to get pid
pid = tl.program_id(axis=0)
```
```
start = pid * BLOCK_SIZE
pid = 0

start = 0 * 256
      = 0
handels: 0-255

pid = 1

start = 1 * 256
      = 256
hendles: 256-511
```

```
calculating offset
tl.arange(0, BLOCK_SIZE) --> [0,1,2,3,4,5,6,7]
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
```

## tl.load(pointer)
```
it reads data from the gpu memory
```
```c
x = torch.tensor(
    [10,20,30,40,50,60,70,80],
    device="cuda"
)
x_ptr = 1000 //starting adress
value = tl.load(x_ptr + 2) // value is 30
```

```c
// loading multiple elements
x = tl.load(x_ptr + offsets)
```

## out of bonds
```
we can go out of the bonds as the ofset may be calculated let say till 1024 but we have actual memory till 1000 then the rest elements are
out of bonds to solve this we use masks

mask = offsets < n_elements
```

```c
x = tl.load(
    x_ptr + offsets,
    mask=mask,
    other=0
)
// the elements that are out of the bonds will be given value 0
```
## tl.store()
```
used to write the data back to the memory
```
```c
// single store value
tl.store(output_ptr + 2, 100)
```
```c
// multiple value write
tl.store(
    output_ptr + offsets,
    output,
    mask=mask
)
```

## Launching triton
### traditional
```c
add_kernel[4](
    x_ptr,
    y_ptr,
    output_ptr,
    N,
    BLOCK_SIZE=256
)

// grid can be 4 or (2,2)
```
```
launches exactly 4 triton program
dynamically calculating number of programs or grid
grid = (
    N + BLOCK_SIZE - 1
) // BLOCK_SIZE
```
### triton style(celing devision)
```c
grid = lambda meta: (
    triton.cdiv(
        N,
        meta['BLOCK_SIZE']
    ),
)
```
