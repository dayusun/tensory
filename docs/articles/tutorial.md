# Tensory Tutorial

## Introduction

Welcome to `tensory`, a modern, high-performance R package for tensor
operations. `tensory` combines a user-friendly R6 class system on the
frontend with a high-performance C++ `xtensor` and Fortran BLAS backend.

The `tensory` package is designed to be highly compatible with the
popular MATLAB Tensor Toolbox API, bringing familiar semantics to R but
with a zero-copy memory integration approach natively optimized for
rapid numerical computations.

This tutorial covers the primary classes and operations available in the
package.

### Setup

First, load the package:

``` r

library(tensory)
#> 
#> Attaching package: 'tensory'
#> The following object is masked from 'package:stats':
#> 
#>     reshape
#> The following object is masked from 'package:utils':
#> 
#>     find
#> The following object is masked from 'package:methods':
#> 
#>     kronecker
#> The following objects are masked from 'package:base':
#> 
#>     %*%, kronecker, scale
```

## The `Tensor` Class

`Tensor` is the core R6 class used for representing multi-dimensional
arrays. It automatically handles dimensions and provides method and
operator overloads for a fluid experience.

### Creating Tensors

You can create tensors from vectors, matrices, arrays, or scalars using
the [`tensor()`](https://dayusun.github.io/tensory/reference/Tensor.md)
constructor function, or directly using `Tensor$new()`.

``` r

# From a vector
t_vec <- tensor(1:5)
print(dim(t_vec$as_array()))
#> [1] 5

# From a matrix
t_mat <- tensor(matrix(1:6, nrow = 2, ncol = 3))
t_mat$print()
#> <Tensor object>
#> A tensor of order 2 with dimensions: 2 x 3

# Specifying dimensions explicitly
t_3d <- tensor(1:24, dims = c(2, 3, 4))
t_3d$print()
#> <Tensor object>
#> A tensor of order 3 with dimensions: 2 x 3 x 4

# Creating tensors of zeros or ones
t_zeros <- zeros(c(2, 2))
t_ones <- ones(c(3, 1, 2))
```

### Accessing Tensor Properties

You can easily extract properties like dimensions or the underlying R
array.

``` r

# Get dimensions
t_3d$dim()
#> [1] 2 3 4

# Get number of dimensions (order)
t_3d$ndims()
#> [1] 3

# Get total number of elements
t_3d$length()
#> [1] 24

# Get the underlying R array
arr <- t_3d$as_array()
str(arr)
#>  num [1:2, 1:3, 1:4] 1 2 3 4 5 6 7 8 9 10 ...

# You can also use standard S3 methods to inspect the tensor data
head(t_3d)
#> , , 1
#> 
#>      [,1] [,2] [,3]
#> [1,]    1    3    5
#> [2,]    2    4    6
#> 
#> , , 2
#> 
#>      [,1] [,2] [,3]
#> [1,]    7    9   11
#> [2,]    8   10   12
#> 
#> , , 3
#> 
#>      [,1] [,2] [,3]
#> [1,]   13   15   17
#> [2,]   14   16   18
#> 
#> , , 4
#> 
#>      [,1] [,2] [,3]
#> [1,]   19   21   23
#> [2,]   20   22   24
tail(t_3d)
#> , , 1
#> 
#>      [,1] [,2] [,3]
#> [1,]    1    3    5
#> [2,]    2    4    6
#> 
#> , , 2
#> 
#>      [,1] [,2] [,3]
#> [1,]    7    9   11
#> [2,]    8   10   12
#> 
#> , , 3
#> 
#>      [,1] [,2] [,3]
#> [1,]   13   15   17
#> [2,]   14   16   18
#> 
#> , , 4
#> 
#>      [,1] [,2] [,3]
#> [1,]   19   21   23
#> [2,]   20   22   24
```

### Reshaping and Squeezing Tensors

You can reshape the tensor in-place or squeeze out singleton dimensions.

``` r

# Reshape to a 4x6 matrix
t_reshaped <- t_3d$clone_tensor()$reshape(c(4, 6))
t_reshaped$dim()
#> [1] 4 6

# Squeeze out singleton dimensions
t_singleton <- tensor(1:6, dims = c(1, 2, 1, 3))
t_singleton$dim()
#> [1] 1 2 1 3

t_squeezed <- t_singleton$squeeze()
t_squeezed$dim()
#> [1] 2 3
```

### Arithmetics and Overloaded Operators

`tensory` implements intuitive S3 operators for element-wise arithmetic
and logical operations, supporting broadcasting with scalars.

``` r

# Element-wise arithmetic
t1 <- tensor(1:4, c(2, 2))
t2 <- tensor(c(1, 0, 0, 1), c(2, 2))

t_add <- t1 + t2
t_add$as_array()
#>      [,1] [,2]
#> [1,]    2    3
#> [2,]    2    5

t_mul <- t1 * 10
t_mul$as_array()
#>      [,1] [,2]
#> [1,]   10   30
#> [2,]   20   40

t_mod <- t1 %% 2
t_mod$as_array()
#>      [,1] [,2]
#> [1,]    1    1
#> [2,]    0    0

# Logical and comparison operations
t_eq <- t1 == t2
t_eq$as_array()
#> [1] 1 0 0 0

t_gt <- t1 > 2
t_gt$as_array()
#> [1] 0 0 1 1

t_not <- !t2
t_not$as_array()
#> [1] 0 1 1 0
```

## The `Tenmat` Class (Matricized Tensors)

A `Tenmat` represents a matricized (or unfolded) tensor. It is widely
used in tensor decompositions. The rows of the matricized tensor
correspond to the modes specified by `rdims`, and the columns correspond
to `cdims`.

### Creating Matricizations

You can matricize a `Tensor` along specific dimensions using
[`tenmat()`](https://dayusun.github.io/tensory/reference/Tenmat.md).

``` r

t_data <- tensor(1:24, c(2, 3, 4))

# Matricize along mode 1 (rdims = 1). The rows of the matrix will correspond
# to the 1st dimension of the tensor.
tm1 <- tenmat(t_data, rdims = 1)
tm1$print()
#> <Tenmat object>
#> A matrix corresponding to a tensor of size 2 x 3 x 4 
#> rindices = [ 1 ] (modes of tensor corresponding to rows)
#> cindices = [ 2 3 ] (modes of tensor corresponding to columns)
#> data =
#>      [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9] [,10] [,11] [,12]
#> [1,]    1    3    5    7    9   11   13   15   17    19    21    23
#> [2,]    2    4    6    8   10   12   14   16   18    20    22    24
dim(as.matrix(tm1))
#> [1]  2 12

# Matricize along mode 3
tm3 <- tenmat(t_data, rdims = 3)
dim(as.matrix(tm3))
#> [1] 4 6

# You can also use specialized column mappings:
# "t" for transpose mapping, "fc" for forward cyclic, "bc" for backward cyclic
tm_fc <- tenmat(t_data, rdims = 2, cdims = "fc")
```

### Matrix Operations and Interpolations

The `Tenmat` objects support addition, subtraction, transposition, and
standard matrix multiplication (`%*%`).

``` r

tmA <- tenmat(tensor(1:6, c(2, 3)), 1)
tmB <- tenmat(tensor(7:12, c(3, 2)), 1)

# Transpose
tmA_t <- t(tmA)

# Matrix multiplication directly yields a newly matched Tenmat
tmC <- tmA %*% tmA_t
tmC$print()
#> <Tenmat object>
#> A matrix corresponding to a tensor of size 2 x 2 
#> rindices = [ 1 ] (modes of tensor corresponding to rows)
#> cindices = [ 2 ] (modes of tensor corresponding to columns)
#> data =
#>      [,1] [,2]
#> [1,]   35   44
#> [2,]   44   56

# Easily convert back to standard R matrices or original Tensor shape
mat_C <- as.matrix(tmC)
t_orig <- tensory::as.tensor(tmC)
```

## Tensor Operations

The core feature of `tensory` is its extremely fast tensor contractions
leveraging C++ xtensor-blas optimizations.

### Tensor Times Matrix (ttm)

[`ttm()`](https://dayusun.github.io/tensory/reference/ttm.md) computes a
tensor times a matrix (or matrices) or vector (or vectors) in one (or
more) dimensions.

``` r

t_base <- tensor(1:24, c(4, 3, 2))

# Multiply a matrix along mode 1
# Matrix needs to have ncol equal to the size of mode 1 (which is 4)
m1 <- matrix(1:8, nrow = 2, ncol = 4)
res1 <- ttm(t_base, m1, mode = 1)
res1$dim() # New dim is [2, 3, 2]
#> [1] 2 3 2

# Multiply multiple matrices along different modes simultaneously
m2 <- matrix(1:6, nrow = 2, ncol = 3)
res2 <- ttm(t_base, list(m1, m2), mode = c(1, 2))
res2$dim() # New dim is [2, 2, 2]
#> [1] 2 2 2

# Multiply by a vector (reduces dimensionality automatically)
v1 <- 1:4
res3 <- ttm(t_base, v1, mode = 1)
res3$dim() # Dimension is reduced to [3, 2]
#> [1] 3 2

# Multiply by multiple vectors (can reduce tensor to a scalar)
v2 <- 1:3
v3 <- 1:2
res4 <- ttm(t_base, list(v1, v2, v3), mode = c(1, 2, 3))
res4$as_array() # Returns a scalar value
#> [1] 2940
```

### Tensor Times Tensor (ttt)

[`ttt()`](https://dayusun.github.io/tensory/reference/ttt.md) computes
the generalized product (outer, inner, or contracted) of two tensors.

``` r

A <- tensor(1:12, c(3, 2, 2))
B <- tensor(1:8, c(2, 2, 2))

# Outer product (no dimensions specified)
A_outer_B <- ttt(A, B)
A_outer_B$dim() # Computes full expansion [3, 2, 2, 2, 2, 2]
#> [1] 3 2 2 2 2 2

# Inner product (contracting over all corresponding dimensions)
A2 <- tensor(1:12, c(3, 2, 2))
A_inner_A2 <- ttt(A, A2, dimsA = 1:3, dimsB = 1:3)
A_inner_A2$dim() # Scalar result
#> integer(0)
A_inner_A2$as_array()
#> [1] 650

# Contracted product along specific dimensions
# Contract mode 2 of A with mode 1 of B
A_contract_B <- ttt(A, B, dimsA = 2, dimsB = 1)
A_contract_B$dim() # Result dimensions: [3, 2, 2, 2]
#> [1] 3 2 2 2
```

## Kruskal and Tucker Tensors

The `tensory` package natively mimics MATLAB’s Tensor Toolbox
capabilities for working with structured tensor decompositions,
specifically Kruskal (`KTensor`) and Tucker (`TTensor`) models.

### KTensor (Kruskal Tensor)

A Kruskal tensor is represented by a set of factor matrices and a vector
of weights.

``` r

# Define weights
lambda <- c(1, 2, 3)

# Define factor matrices
U1 <- matrix(runif(12), nrow=4, ncol=3)
U2 <- matrix(runif(15), nrow=5, ncol=3)
U3 <- matrix(runif(6), nrow=2, ncol=3)

k <- ktensor(lambda, list(U1, U2, U3))
print(k)
#> <KTensor object>
#> Dimensions:  4 x 5 x 2 
#> Weights (lambda):  1 2 3 
#> Factor matrices (U): 
#>  U[[1]]: 4 x 3
#>  U[[2]]: 5 x 3
#>  U[[3]]: 2 x 3

# Advanced Mode Multiplication Native Support
# Multiply Kruskal Tensor mode 2 by a vector (reduces dimensionality sequentially)
k_vec <- ttm(k, runif(5), mode = 2)
k_vec$dim() # Factor 2 absorbed, yielding [4, 2]
#> [1] 4 2

# Convert to a Dense Tensor instantly
k_full <- as.tensor(k)
k_full$dim() # [4, 5, 2]
#> [1] 4 5 2
```

### TTensor (Tucker Tensor)

A Tucker tensor represents a dense core tensor scaled factorially by
multiple matrices alongside each corresponding dimension.

``` r

# Define core tensor
core <- tensor(array(runif(24), dim=c(2,3,4)))

# Define structural factor matrices bridging the dimensional mapping
U1_t <- matrix(runif(10), nrow=5, ncol=2)
U2_t <- matrix(runif(18), nrow=6, ncol=3)
U3_t <- matrix(runif(28), nrow=7, ncol=4)

t_tens <- ttensor(core, list(U1_t, U2_t, U3_t))
print(t_tens)
#> <TTensor object>
#> Dimensions:  5 x 6 x 7 
#> Core tensor (core):
#>    2 x 3 x 4  Dense Tensor
#> Factor matrices (U): 
#>  U[[1]]: 5 x 2
#>  U[[2]]: 6 x 3
#>  U[[3]]: 7 x 4

# Advanced Mode Multiplication Native Support
# Multiply Tucker Tensor mode 1 by a matrix directly without instantiation
t_mat <- matrix(runif(15), nrow=3, ncol=5)
t_tens_mat <- ttm(t_tens, t_mat, mode = 1)
t_tens_mat$dim() # Core factor 1 updates dimensions efficiently to [3, 6, 7]
#> [1] 3 6 7

# Multiply Tucker Tensor natively across multiple modes via sequential matrices
t_list_mats <- list(matrix(runif(12), nrow=4, ncol=5), matrix(runif(10), nrow=2, ncol=6))
#> Warning in matrix(runif(12), nrow = 4, ncol = 5): data length [12] is not a
#> sub-multiple or multiple of the number of columns [5]
#> Warning in matrix(runif(10), nrow = 2, ncol = 6): data length [10] is not a
#> sub-multiple or multiple of the number of columns [6]
t_tens_list <- ttm(t_tens, t_list_mats, mode = c(1, 2))
t_tens_list$dim() # Sequentially updates to [4, 2, 7] efficiently
#> [1] 4 2 7

# Convert to a Dense Tensor instantly
t_full <- as.tensor(t_tens)
t_full$dim() # [5, 6, 7]
#> [1] 5 6 7
```

### Seamless Mathematical Dispatch Operations

Because of `tensory`’s fluid S3 dispatch ecosystem, `KTensor` and
`TTensor` objects play entirely nicely with standard multiplication
(`ttm`, `ttt`) and base math operators by implicitly casting to dense
components smoothly on-the-fly.

``` r

# Add compatible ktensor and ttensor naturally
compat_core <- tensor(array(runif(40), dim = c(4, 5, 2)))
compat_tens <- ttensor(compat_core, list(diag(4), diag(5), diag(2)))
t_add <- k + compat_tens
t_add$dim()
#> [1] 4 5 2

# Generalized Outer product
k_ttt <- ttt(k, t_tens)
k_ttt$dim()
#> [1] 4 5 2 5 6 7
```

## Advanced Mathematical Operations

`tensory` implements specialized matrix and tensor operations leveraging
S3 generics for multiple-dispatch (e.g., smoothly operating on pairs of
matrices/tensors or even lists of matrices). Most of these operations
are also available as direct R6 methods on the `Tensor` objects.

### Khatri-Rao, Kronecker, and Hadamard

You can easily compute these specialized products. If extending over a
list of matrices, the functions recursively apply the product.

``` r

matA <- matrix(1:4, nrow = 2, ncol = 2)
matB <- matrix(5:8, nrow = 2, ncol = 2)

# Khatri-Rao Product (column-wise Kronecker)
khatri_rao(matA, matB)
#>      [,1] [,2]
#> [1,]    5   21
#> [2,]    6   24
#> [3,]   10   28
#> [4,]   12   32

# By default it is (matA \odot matB).
# You can reverse this to (matB \odot matA)
khatri_rao(matA, matB, reverse = TRUE)
#>      [,1] [,2]
#> [1,]    5   21
#> [2,]   10   28
#> [3,]    6   24
#> [4,]   12   32

# Kronecker Product (extends base::kronecker)
kronecker(list(matA, matB))
#>      [,1] [,2] [,3] [,4]
#> [1,]    5    7   15   21
#> [2,]    6    8   18   24
#> [3,]   10   14   20   28
#> [4,]   12   16   24   32

# Hadamard Product (element-wise multiplication)
hadamard(list(matA, matB))
#>      [,1] [,2]
#> [1,]    5   21
#> [2,]   12   32
```

### Frobenius Norm

You can calculate the full Frobenius norm of a tensor directly.

``` r

t_norm <- tensor(c(3, 4))
fnorm(t_norm) # Output: 5
#> [1] 5

# Or via R6 method
t_norm$fnorm()
#> [1] 5
```

### Collapsing and Scaling

[`collapse()`](https://dayusun.github.io/tensory/reference/collapse.md)
reduces a tensor along specific dimensions using an accumulation
function, while
[`t_scale()`](https://dayusun.github.io/tensory/reference/t_scale.md)
performs dimension-aligned multiplying (broadcasting).

``` r

t_dense <- tensor(1:24, c(4, 3, 2))

# Collapse Mode 1 (Sum all elements across rows)
tensory::collapse(t_dense, 1)$dim()
#> [1] 3 2

# Collapse all modes EXCEPT Mode 3
# Negative dimensions mean "exclude these dimensions from collapse"
tensory::collapse(t_dense, -3)$dim()
#> [1] 2

# Scale mode 3 with a specific vector
t_scale(t_dense, c(10, 20), dims = 3)$dim()
#> [1] 4 3 2
```

## Conclusion

The `tensory` package brings the flexibility of MATLAB Tensor Toolbox to
the R ecosystem with heavily optimized `xtensor` bindings and an elegant
R6 object-oriented interface. You can create tensors, matricize them
mathematically, apply advanced reductions and element-wise scalings, and
compute multi-dimensional products flawlessly.
