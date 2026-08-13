# R6 Tensor Class

A modern tensor class for R that provides MATLAB Tensor Toolbox
compatibility with high-performance operations via xtensor C++ backend.

Convenience function to create a Tensor object

## Usage

``` r
tensor(data, dims = NULL)
```

## Arguments

- data:

  A vector, matrix, array, or numeric value

- dims:

  The dimensions of the tensor. If NULL, inferred from data

## Value

A new Tensor object

## Public fields

- `data`:

  The underlying array data

- `dims`:

  The dimensions of the tensor Initialize a new tensor

## Methods

### Public methods

- [`Tensor$new()`](#method-Tensor-initialize)

- [`Tensor$dim()`](#method-Tensor-dim)

- [`Tensor$length()`](#method-Tensor-length)

- [`Tensor$ndims()`](#method-Tensor-ndims)

- [`Tensor$as_array()`](#method-Tensor-as_array)

- [`Tensor$print()`](#method-Tensor-print)

- [`Tensor$show()`](#method-Tensor-show)

- [`Tensor$clone_tensor()`](#method-Tensor-clone_tensor)

- [`Tensor$reshape()`](#method-Tensor-reshape)

- [`Tensor$squeeze()`](#method-Tensor-squeeze)

- [`Tensor$permute()`](#method-Tensor-permute)

- [`Tensor$nnz()`](#method-Tensor-nnz)

- [`Tensor$find()`](#method-Tensor-find)

- [`Tensor$vec()`](#method-Tensor-vec)

- [`Tensor$add()`](#method-Tensor-add)

- [`Tensor$subtract()`](#method-Tensor-subtract)

- [`Tensor$multiply()`](#method-Tensor-multiply)

- [`Tensor$divide()`](#method-Tensor-divide)

- [`Tensor$modulo()`](#method-Tensor-modulo)

- [`Tensor$integer_divide()`](#method-Tensor-integer_divide)

- [`Tensor$power()`](#method-Tensor-power)

- [`Tensor$equal()`](#method-Tensor-equal)

- [`Tensor$not_equal()`](#method-Tensor-not_equal)

- [`Tensor$less_than()`](#method-Tensor-less_than)

- [`Tensor$less_equal()`](#method-Tensor-less_equal)

- [`Tensor$greater_than()`](#method-Tensor-greater_than)

- [`Tensor$greater_equal()`](#method-Tensor-greater_equal)

- [`Tensor$logical_not()`](#method-Tensor-logical_not)

- [`Tensor$logical_and()`](#method-Tensor-logical_and)

- [`Tensor$logical_or()`](#method-Tensor-logical_or)

- [`Tensor$sum()`](#method-Tensor-sum)

- [`Tensor$khatri_rao()`](#method-Tensor-khatri_rao)

- [`Tensor$kronecker()`](#method-Tensor-kronecker)

- [`Tensor$hadamard()`](#method-Tensor-hadamard)

- [`Tensor$fnorm()`](#method-Tensor-fnorm)

- [`Tensor$clone()`](#method-Tensor-clone)

------------------------------------------------------------------------

### `Tensor$new()`

#### Usage

    Tensor$new(data = NULL, dims = NULL, fast = FALSE)

#### Arguments

- `data`:

  A vector, matrix, array, or numeric value

- `dims`:

  The dimensions of the tensor. If NULL, inferred from data

- `fast`:

  If TRUE, bypasses all checks. Used internally for performance.

#### Returns

A new Tensor object Get tensor dimensions

------------------------------------------------------------------------

### `Tensor$dim()`

#### Usage

    Tensor$dim()

#### Returns

Integer vector of dimensions Get number of elements

------------------------------------------------------------------------

### `Tensor$length()`

#### Usage

    Tensor$length()

#### Returns

Integer number of elements Get number of dimensions

------------------------------------------------------------------------

### `Tensor$ndims()`

#### Usage

    Tensor$ndims()

#### Returns

Integer number of dimensions Convert to R array

------------------------------------------------------------------------

### `Tensor$as_array()`

#### Usage

    Tensor$as_array()

#### Returns

R array representation Print tensor information

------------------------------------------------------------------------

### `Tensor$print()`

#### Usage

    Tensor$print()

#### Returns

Invisible self Show tensor (alias for print)

------------------------------------------------------------------------

### `Tensor$show()`

#### Usage

    Tensor$show()

#### Returns

Invisible self Clone the tensor

------------------------------------------------------------------------

### `Tensor$clone_tensor()`

#### Usage

    Tensor$clone_tensor()

#### Returns

A new Tensor object with copied data Reshape the tensor

------------------------------------------------------------------------

### `Tensor$reshape()`

#### Usage

    Tensor$reshape(new_dims)

#### Arguments

- `new_dims`:

  New dimensions

#### Returns

Self (in-place operation) Squeeze the tensor by removing singleton
dimensions

------------------------------------------------------------------------

### `Tensor$squeeze()`

#### Usage

    Tensor$squeeze()

#### Returns

New Tensor object with singleton dimensions removed Permute tensor
dimensions

------------------------------------------------------------------------

### `Tensor$permute()`

#### Usage

    Tensor$permute(order)

#### Arguments

- `order`:

  Permutation order

#### Returns

New Tensor object Count nonzero entries

------------------------------------------------------------------------

### `Tensor$nnz()`

#### Usage

    Tensor$nnz()

#### Returns

Integer count Find nonzero entries

------------------------------------------------------------------------

### `Tensor$find()`

#### Usage

    Tensor$find(values = FALSE)

#### Arguments

- `values`:

  Logical; include values if TRUE

#### Returns

Matrix of subscripts or list with subs and vals Vectorize tensor

------------------------------------------------------------------------

### `Tensor$vec()`

#### Usage

    Tensor$vec()

#### Returns

Numeric vector Element-wise addition

------------------------------------------------------------------------

### `Tensor$add()`

#### Usage

    Tensor$add(other)

#### Arguments

- `other`:

  Another Tensor or numeric value

#### Returns

Self (in-place operation) Element-wise subtraction

------------------------------------------------------------------------

### `Tensor$subtract()`

#### Usage

    Tensor$subtract(other)

#### Arguments

- `other`:

  Another Tensor or numeric value

#### Returns

Self (in-place operation) Element-wise multiplication

------------------------------------------------------------------------

### `Tensor$multiply()`

#### Usage

    Tensor$multiply(other)

#### Arguments

- `other`:

  Another Tensor or numeric value

#### Returns

Self (in-place operation) Element-wise division

------------------------------------------------------------------------

### `Tensor$divide()`

#### Usage

    Tensor$divide(other)

#### Arguments

- `other`:

  Another Tensor or numeric value

#### Returns

Self (in-place operation) Element-wise modulo

------------------------------------------------------------------------

### `Tensor$modulo()`

#### Usage

    Tensor$modulo(other)

#### Arguments

- `other`:

  Another Tensor or numeric value

#### Returns

Self (in-place operation) Element-wise integer division

------------------------------------------------------------------------

### `Tensor$integer_divide()`

#### Usage

    Tensor$integer_divide(other)

#### Arguments

- `other`:

  Another Tensor or numeric value

#### Returns

Self (in-place operation) Element-wise power

------------------------------------------------------------------------

### `Tensor$power()`

#### Usage

    Tensor$power(other)

#### Arguments

- `other`:

  Another Tensor or numeric value

#### Returns

Self (in-place operation) Element-wise equality comparison

------------------------------------------------------------------------

### `Tensor$equal()`

#### Usage

    Tensor$equal(other)

#### Arguments

- `other`:

  Another Tensor or numeric value

#### Returns

Self (in-place operation) Element-wise inequality comparison

------------------------------------------------------------------------

### `Tensor$not_equal()`

#### Usage

    Tensor$not_equal(other)

#### Arguments

- `other`:

  Another Tensor or numeric value

#### Returns

Self (in-place operation) Element-wise less than comparison

------------------------------------------------------------------------

### `Tensor$less_than()`

#### Usage

    Tensor$less_than(other)

#### Arguments

- `other`:

  Another Tensor or numeric value

#### Returns

Self (in-place operation) Element-wise less than or equal comparison

------------------------------------------------------------------------

### `Tensor$less_equal()`

#### Usage

    Tensor$less_equal(other)

#### Arguments

- `other`:

  Another Tensor or numeric value

#### Returns

Self (in-place operation) Element-wise greater than comparison

------------------------------------------------------------------------

### `Tensor$greater_than()`

#### Usage

    Tensor$greater_than(other)

#### Arguments

- `other`:

  Another Tensor or numeric value

#### Returns

Self (in-place operation) Element-wise greater than or equal comparison

------------------------------------------------------------------------

### `Tensor$greater_equal()`

#### Usage

    Tensor$greater_equal(other)

#### Arguments

- `other`:

  Another Tensor or numeric value

#### Returns

Self (in-place operation) Element-wise logical NOT

------------------------------------------------------------------------

### `Tensor$logical_not()`

#### Usage

    Tensor$logical_not()

#### Returns

Self (in-place operation) Element-wise logical AND

------------------------------------------------------------------------

### `Tensor$logical_and()`

#### Usage

    Tensor$logical_and(other)

#### Arguments

- `other`:

  Another Tensor or numeric value

#### Returns

Self (in-place operation) Element-wise logical OR

------------------------------------------------------------------------

### `Tensor$logical_or()`

#### Usage

    Tensor$logical_or(other)

#### Arguments

- `other`:

  Another Tensor or numeric value

#### Returns

Self (in-place operation) Sum along dimensions

------------------------------------------------------------------------

### `Tensor$sum()`

#### Usage

    Tensor$sum(dims = NULL)

#### Arguments

- `dims`:

  Dimensions to sum along (NULL for all)

#### Returns

New Tensor with reduced dimensions Khatri-Rao product with another
Tensor or matrix

------------------------------------------------------------------------

### `Tensor$khatri_rao()`

#### Usage

    Tensor$khatri_rao(other, reverse = FALSE)

#### Arguments

- `other`:

  Another Tensor or matrix

- `reverse`:

  Logical indicating if reverse product should be computed

#### Returns

A matrix representing the Khatri-Rao product Kronecker product with
another Tensor or matrix

------------------------------------------------------------------------

### `Tensor$kronecker()`

#### Usage

    Tensor$kronecker(other)

#### Arguments

- `other`:

  Another Tensor or matrix

#### Returns

A new Tensor Hadamard (element-wise) product

------------------------------------------------------------------------

### `Tensor$hadamard()`

#### Usage

    Tensor$hadamard(other)

#### Arguments

- `other`:

  Another Tensor or matrix

#### Returns

Self (in-place operation) Frobenius norm

------------------------------------------------------------------------

### `Tensor$fnorm()`

#### Usage

    Tensor$fnorm()

#### Returns

Numeric scalar

------------------------------------------------------------------------

### `Tensor$clone()`

The objects of this class are cloneable with this method.

#### Usage

    Tensor$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
# Create a tensor from a matrix
t <- Tensor$new(matrix(1:6, nrow = 2, ncol = 3))
print(t)
#> <Tensor object>
#> A tensor of order 2 with dimensions: 2 x 3 

# Create a tensor with specific dimensions
t2 <- Tensor$new(1:24, c(2, 3, 4))

# Mathematical operations using method syntax
t3 <- t2$clone_tensor()$add(t2)

# Mathematical operations using operator syntax
t4 <- t2 + t2
t5 <- t2 * 2
t6 <- 5 - t2
t7 <- t2 / 3

# Create a tensor from an array
t1 <- tensor(array(1:24, dim = c(3, 4, 2)))

# Create a tensor by specifying dimensions
t2 <- tensor(1:24, dims = c(3, 4, 2))

# Create a tensor filled with a single value
t3 <- tensor(0, dims = c(3, 4, 2))
```
