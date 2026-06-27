# Triton Matrix Multiplication Explained from Scratch

> A beginner-friendly intuition guide to understand how Triton performs Matrix Multiplication (MatMul) on GPUs.

---

# Table of Contents

1. Why Matrix Multiplication?
2. Matrix Multiplication Basics
3. Why CPUs are Slow for MatMul
4. Why GPUs are Fast
5. GPU Memory Hierarchy
6. How Tensors are Stored in Memory
7. Pointer Arithmetic
8. Triton Execution Model
9. Tiling
10. Computing Tile Coordinates
11. Loading Data (`tl.load`)
12. Matrix Multiplication using `tl.dot`
13. The K Loop
14. Accumulator
15. Writing the Output (`tl.store`)
16. Complete Execution Flow
17. Key Takeaways

---

# 1. Why Matrix Multiplication?

Almost every Deep Learning model spends most of its computation doing matrix multiplication.

Examples:

- Linear Layers
- CNNs
- Transformers
- Attention
- MLPs

Everything eventually becomes

```
C = A × B
```

Optimizing Matrix Multiplication means optimizing Deep Learning.

---

# 2. Matrix Multiplication Basics

Suppose

```
A

1 2
3 4
```

```
B

5 6
7 8
```

Output

```
C = A × B
```

Each output element is

```
Row of A

×

Column of B
```

Example

```
C00

=

1×5 + 2×7

=

19
```

General Formula

```
C[i][j] = Σ A[i][k] × B[k][j]
```

Notice that we sum over **K**.

---

# 3. Why CPUs are Slow

A CPU computes something like

```cpp
for(i)
    for(j)
        for(k)
            C[i][j] += A[i][k] * B[k][j];
```

This works well for small matrices.

But for large matrices

```
4096 × 4096
```

there are billions of operations.

A CPU has only a few cores.

---

# 4. Why GPUs are Fast

A GPU has thousands of cores.

Instead of computing

```
One output
```

it computes many outputs simultaneously.

Example

```
□□□□

□□□□

□□□□

□□□□
```

Each square is one output element.

Thousands of these are computed in parallel.

---

# 5. GPU Memory Hierarchy

```
        Registers
             ↑
     Shared Memory
             ↑
         L1 Cache
             ↑
         L2 Cache
             ↑
     Global Memory (VRAM)
```

### Registers

- Fastest memory
- Stores temporary variables
- Used during computation

### Shared Memory

- Faster than Global Memory
- Shared inside a thread block
- Often used to reuse data

### Cache

Automatically stores recently accessed data.

### Global Memory

- Stores tensors
- Largest memory
- Slowest memory

PyTorch tensors live here.

---

# 6. How Tensors are Stored

Suppose

```
Tensor

1 2 3

4 5 6
```

Memory actually looks like

```
1000 → 1

1004 → 2

1008 → 3

1012 → 4

1016 → 5

1020 → 6
```

Everything is stored linearly.

A pointer stores the address of the first element.

---

# 7. Pointer Arithmetic

Suppose

```
Base Address = 1000
```

To reach

```
Row = 2

Column = 1
```

we compute

```
Address

=

Base

+

Row × RowStride

+

Column × ColumnStride
```

This is exactly how Triton computes addresses.

---

# 8. Triton Execution Model

Instead of thinking

```
Thread

↓

One Element
```

Triton thinks

```
Program

↓

One Tile
```

Every Triton Program computes one block of the output matrix.

Example

```
□□□□□□□□

□□□□□□□□

□□□□□□□□

□□□□□□□□
```

---

# 9. Tiling

Instead of multiplying the entire matrix at once,

split it into smaller pieces.

Example

```
Large Matrix

□□□□□□□□□□□□□□□□□□□□
□□□□□□□□□□□□□□□□□□□□
□□□□□□□□□□□□□□□□□□□□
□□□□□□□□□□□□□□□□□□□□
```

becomes

```
■■■■

■■■■

■■■■

■■■■
```

Each square is called a **Tile**.

Benefits

- Less memory traffic
- Better cache usage
- Data reuse

---

# 10. Computing Tile Coordinates

Suppose

```
BLOCK_M = 128

BLOCK_N = 128
```

Each program computes

```
128 × 128
```

output elements.

The program first finds

```
Which tile do I own?
```

using

```python
pid = tl.program_id(0)
```

Then computes

```
Rows

Columns
```

belonging to that tile.

---

# 11. Loading Data (`tl.load`)

The most important instruction

```python
a = tl.load(a_ptrs)
```

What happens?

```
Global Memory

↓

tl.load()

↓

Registers
```

After loading

```
A Tile
```

is available inside the GPU registers.

The same happens for B.

```
Global Memory

↓

Registers
```

Computation happens **only after loading**.

---

# 12. Matrix Multiplication (`tl.dot`)

Suppose

```
A Tile

128 × 32
```

and

```
B Tile

32 × 128
```

Then

```python
partial = tl.dot(a, b)
```

computes

```
128 × 128
```

partial output.

Think of

```python
tl.dot()
```

as

```
torch.matmul()
```

but operating on small tiles.

---

# 13. The K Loop

A matrix may have

```
K = 4096
```

columns.

We cannot load everything.

Instead

```
Load Tile

↓

Multiply

↓

Accumulate

↓

Load Next Tile

↓

Multiply

↓

Accumulate
```

Pseudo Code

```python
for k in range(0, K, BLOCK_K):

    a = tl.load(...)

    b = tl.load(...)

    accumulator += tl.dot(a, b)
```

The loop continues until the entire K dimension is processed.

---

# 14. The Accumulator

Initially

```
0 0 0

0 0 0

0 0 0
```

After first iteration

```
5 6 7

8 9 10
```

After second iteration

```
15 16 17

18 19 20
```

The accumulator stores the running sum.

Usually it is kept in **Registers**.

---

# 15. Writing the Output (`tl.store`)

Once computation finishes

```python
tl.store(c_ptrs, accumulator)
```

Data moves

```
Registers

↓

Global Memory
```

The output tensor is now stored in VRAM.

---

# 16. Complete Execution Flow

```
PyTorch Tensor

↓

Global Memory

↓

tl.load()

↓

Registers

↓

tl.dot()

↓

Accumulator

↓

tl.store()

↓

Global Memory

↓

Output Tensor
```

---

# 17. Visualizing the Entire MatMul

```
Matrix A

■■■■□□□□

■■■■□□□□

■■■■□□□□

■■■■□□□□


Matrix B

■■■■

■■■■

■■■■

■■■■

□□□□

□□□□


↓

Load Tiles

↓

Registers

↓

tl.dot()

↓

Accumulator

↓

Next Tile

↓

Accumulator += tl.dot()

↓

Repeat until K finishes

↓

tl.store()

↓

Output Tile
```

---

# Key Takeaways

- Matrix Multiplication is the foundation of Deep Learning.
- GPUs speed up MatMul by computing many outputs in parallel.
- Triton computes one **tile** per program instead of one element.
- `tl.load()` copies data from **Global Memory → Registers**.
- `tl.dot()` multiplies two small matrix tiles.
- The K loop processes the matrix in smaller chunks.
- The accumulator stores partial sums in registers.
- `tl.store()` writes the final result back to Global Memory.
- Tiling improves performance by reducing memory accesses and reusing data.
- The overall flow is:

```
Global Memory
      ↓
  tl.load()
      ↓
  Registers
      ↓
   tl.dot()
      ↓
Accumulator
      ↓
  tl.store()
      ↓
Global Memory
```

---

# Mental Model

Whenever you see a Triton MatMul kernel, think:

```
Find My Tile
        ↓
Compute Tile Coordinates
        ↓
Load Tile A
        ↓
Load Tile B
        ↓
Multiply Tiles
        ↓
Accumulate
        ↓
Move to Next K Tile
        ↓
Repeat
        ↓
Store Final Tile
```

If you understand this flow, you understand the core idea behind Triton's matrix multiplication.