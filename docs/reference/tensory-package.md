# tensory: Tensory - Modern Tensor Operations for R

Provides dense and decomposed tensor classes for R together with
operations that use an API similar to the MATLAB Tensor Toolbox,
including matricized products, contractions, reshaping helpers, and
symmetry utilities. Selected kernels can delegate to
\`xtensor\`/\`Rcpp\` backends for performance.

`tensory` provides dense and decomposed tensor classes together with an
API similar to the MATLAB Tensor Toolbox for R users.

## Details

The `tensory` package provides tensor algebra tools for R with an API
similar to the MATLAB Tensor Toolbox. It includes dense tensors,
matricized tensors, and decomposed tensor representations, with selected
operations able to delegate to optimized C++ backends.

## Key Features

- R6-based tensor classes for dense and decomposed representations

- Operations with names and semantics similar to the MATLAB Tensor
  Toolbox such as
  [`ttm()`](https://www.sundayu.me/tensory/reference/ttm.md),
  [`ttt()`](https://www.sundayu.me/tensory/reference/ttt.md),
  [`ttv()`](https://www.sundayu.me/tensory/reference/ttv.md),
  [`permute()`](https://www.sundayu.me/tensory/reference/permute.md),
  and
  [`contract()`](https://www.sundayu.me/tensory/reference/contract.md)

- Dense helpers for unfolding, vectorization, symmetry checks, and
  matricized tensor products

- Optional C++ acceleration hooks via `xtensor` and `Rcpp`

## Main Classes

- [`Tensor`](https://www.sundayu.me/tensory/reference/Tensor.md): The
  core R6 class for tensor operations

- [`Tenmat`](https://www.sundayu.me/tensory/reference/Tenmat.md):
  Matricized tensor representation

- [`KTensor`](https://www.sundayu.me/tensory/reference/KTensor.md):
  Kruskal tensor representation

- [`TTensor`](https://www.sundayu.me/tensory/reference/TTensor.md):
  Tucker tensor representation

## Main Functions

- [`tensor`](https://www.sundayu.me/tensory/reference/Tensor.md): Create
  tensor objects from R data structures

- [`zeros`](https://www.sundayu.me/tensory/reference/zeros.md): Create
  tensors filled with zeros

- [`ones`](https://www.sundayu.me/tensory/reference/ones.md): Create
  tensors filled with ones

- [`ttm`](https://www.sundayu.me/tensory/reference/ttm.md): Tensor times
  matrix or vector

- [`ttt`](https://www.sundayu.me/tensory/reference/ttt.md):
  Tensor-times-tensor products

- [`permute`](https://www.sundayu.me/tensory/reference/permute.md):
  Reorder tensor dimensions

- [`mttkrp`](https://www.sundayu.me/tensory/reference/mttkrp.md):
  Matricized tensor times Khatri-Rao product

## Performance

The package is structured so dense operations can use optimized C++
implementations when available while preserving an R-friendly interface
and predictable tensor shapes.

## Getting Started


    # Create a tensor from a matrix
    t <- tensor(matrix(1:6, nrow=2, ncol=3))
    print(t)

    # Create a 3D tensor
    t3d <- tensor(1:24, c(2, 3, 4))

    # Mathematical operations
    result <- t3d$clone_tensor()$add(t3d)$multiply(2)

## Author

Dayu Sun
