# Lesson: Optimizing Tensor-Times-Matrix (TTM) in C++

## The Problem with Transpose

When computing a Tensor-Times-Matrix (TTM) multiplication along a specific mode (dimension) $k$, the standard approach often involves:

1. **Transposing** the tensor to bring mode $k$ to the front (or back).
2. **Reshaping** the tensor into a 2D matrix.
3. Performing a standard **Matrix Multiplication** via BLAS (`dgemm`).
4. **Reshaping** and **Transposing** the result back to the correct original dimension ordering.

While mathematically simple, transposing large tensors requires moving almost every single element to a new memory location. This memory allocation and copying (which happens twice—once before and once after the multiplication) is heavily memory-bound and often takes longer than the actual arithmetic of the matrix multiplication!

## How We Avoided Transpose: Sliced DGEMM

To understand how we skipped the transpose step completely, we have to look at how data is laid out in memory.

R uses **Column-Major Order**. This means the first dimension varies fastest in memory, the second dimension varies next fastest, and so on.

Imagine a tensor $X$ with dimensions $(I_1, I_2, \dots, I_k, \dots, I_N)$. We want to contract it along mode $k$ with a matrix $M$.

We can conceptually flatten the tensor into exactly 3 dimensions: $M_1 \times I_k \times M_2$, where:

- **$M_1$** = $I_1 \times I_2 \times \dots \times I_{k-1}$ (Product of dimensions _before_ mode $k$)
- **$I_k$** = Size of the dimension we are contracting
- **$M_2$** = $I_{k+1} \times \dots \times I_N$ (Product of dimensions _after_ mode $k$)

Because of column-major ordering, for any fixed index in the $M_2$ dimensions, the block of memory representing $M_1 \times I_k$ is exactly contiguous in memory. It forms a perfect $M_1 \times I_k$ matrix!

### The Loop Strategy

Instead of moving memory around to bring $k$ to the front, we simply iterate over the $M_2$ outer slices.

For each slice $m_2$ (where $m_2$ goes from $0$ to $M_2-1$):

1. **Take a pointer** to the start of the slice in the input tensor $X$. This points to a matrix of size $M_1 \times I_k$.
2. We want to multiply this $M_1 \times I_k$ matrix by our matrix $M$ (which is $J \times I_k$ in the normal un-transposed case).
3. The math for this slice is $Y_{slice} = X_{slice} \times M^T$.
4. **Take a pointer** to the start of the slice in our output tensor $Y$. This points to where we will write a matrix of size $M_1 \times J$.
5. Call `dgemm` (BLAS). We tell BLAS to compute $A \times B^T$, where $A$ is our slice ($M_1 \times I_k$) and $B$ is matrix $M$ ($J \times I_k$).

Since BLAS functions like `dgemm` can transpose matrices on the fly without making copies (just by changing how they read the strides), the entire operation for that slice is done natively in CPU cache. We write the result directly into the correct spot in the output tensor! **Zero reshaping, zero memory transposition.**

### The Special Case for Mode 1 ($M_1 = 1$)

If we are contracting along the very first dimension (Mode 1), then $I_k$ is the first dimension. There are no dimensions before it, so $M_1 = 1$.

The tensor is essentially $I_1 \times M_2$.
The matrix $M$ is $J \times I_1$.
The output $Y$ is $J \times M_2$.

We don't even need a loop! The entire operation reduces to a single matrix multiplication: $Y = M \times X_{flattened}$.

We simply pass the pointer for the raw tensor data and the raw matrix data to a single `dgemm` call, completely skipping the $M_2$ loop. This drastically speeds up Mode 1 calculations by removing the overhead of calling `dgemm` thousands of times for tiny slices.

## What About General Tensors (e.g., 5 Modes)?

The beauty of this optimization is that it scales to **any number of dimensions** without changing the logic!

Whether the tensor has 3 modes or 5 modes (e.g., $I_1 \times I_2 \times I_3 \times I_4 \times I_5$), the exact same zero-copy trick works seamlessly for any mode $k$.

For example, if we are contracting along **Mode 3** ($I_3$):

- **$M_1$**: The product of all dimensions _before_ Mode 3 ($I_1 \times I_2$).
- **$I_k$**: The dimension of Mode 3 itself ($I_3$).
- **$M_2$**: The product of all dimensions _after_ Mode 3 ($I_4 \times I_5$).

Because of the column-major memory layout, the computer's memory doesn't care whether the $M_1$ elements came from 1 mode or 20 modes—they still form a perfectly contiguous memory block of length $M_1$. Similarly, $M_2$ merely reflects how many times we must repeat the matrix multiplication to reach the end of the tensor.

Thus, any $N$-dimensional tensor operation logically collapses back down into the exact same 3-dimensional flattened problem: $M_1 \times I_k \times M_2$. We just iterate $M_2$ times, grab the contiguous $M_1 \times I_k$ memory slices, and let `dgemm` multiply them!
