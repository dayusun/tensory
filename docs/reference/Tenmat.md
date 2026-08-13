# R6 Tenmat Class

A class representing a matricized tensor, equivalent to MATLAB Tensor
Toolbox's tenmat.

## Usage

``` r
tenmat(T, rdims = NULL, cdims = NULL, tsize = NULL)
```

## Arguments

- T:

  A Tensor object, matrix, or Tenmat object

- rdims:

  Dimensions to map to rows

- cdims:

  Dimensions to map to columns (optional). Also accepts 't', 'fc', 'bc'.

- tsize:

  Tensor size (used internally)

## Value

A new Tenmat object

## Public fields

- `data`:

  The underlying 2D matrix data

- `rdims`:

  The modes mapped to the rows

- `cdims`:

  The modes mapped to the columns

- `tsize`:

  The size of the original tensor Initialize a new tenmat

## Methods

### Public methods

- [`Tenmat$new()`](#method-Tenmat-initialize)

- [`Tenmat$dim()`](#method-Tenmat-dim)

- [`Tenmat$length()`](#method-Tenmat-length)

- [`Tenmat$print()`](#method-Tenmat-print)

- [`Tenmat$show()`](#method-Tenmat-show)

- [`Tenmat$clone_tenmat()`](#method-Tenmat-clone_tenmat)

- [`Tenmat$add()`](#method-Tenmat-add)

- [`Tenmat$subtract()`](#method-Tenmat-subtract)

- [`Tenmat$negate()`](#method-Tenmat-negate)

- [`Tenmat$transpose()`](#method-Tenmat-transpose)

- [`Tenmat$clone()`](#method-Tenmat-clone)

------------------------------------------------------------------------

### `Tenmat$new()`

#### Usage

    Tenmat$new(data = NULL, rdims = NULL, cdims = NULL, tsize = NULL)

#### Arguments

- `data`:

  The matrix representation

- `rdims`:

  The row indices

- `cdims`:

  The column indices

- `tsize`:

  The size of the original tensor

#### Returns

A new Tenmat object Get dimension of the underlying matrix

------------------------------------------------------------------------

### `Tenmat$dim()`

#### Usage

    Tenmat$dim()

#### Returns

Integer vector of dimensions Get number of elements

------------------------------------------------------------------------

### `Tenmat$length()`

#### Usage

    Tenmat$length()

#### Returns

Integer number of elements Print tenmat information

------------------------------------------------------------------------

### `Tenmat$print()`

#### Usage

    Tenmat$print()

#### Returns

Invisible self Show tenmat (alias for print)

------------------------------------------------------------------------

### `Tenmat$show()`

#### Usage

    Tenmat$show()

#### Returns

Invisible self Clone the tenmat

------------------------------------------------------------------------

### `Tenmat$clone_tenmat()`

#### Usage

    Tenmat$clone_tenmat()

#### Returns

A new Tenmat object Add another Tenmat

------------------------------------------------------------------------

### `Tenmat$add()`

#### Usage

    Tenmat$add(other)

#### Arguments

- `other`:

  Another Tenmat object

#### Returns

A new Tenmat object Subtract another Tenmat

------------------------------------------------------------------------

### `Tenmat$subtract()`

#### Usage

    Tenmat$subtract(other)

#### Arguments

- `other`:

  Another Tenmat object

#### Returns

A new Tenmat object Negate Tenmat

------------------------------------------------------------------------

### `Tenmat$negate()`

#### Usage

    Tenmat$negate()

#### Returns

A new Tenmat object Transpose Tenmat

------------------------------------------------------------------------

### `Tenmat$transpose()`

#### Usage

    Tenmat$transpose()

#### Returns

A new Tenmat object

------------------------------------------------------------------------

### `Tenmat$clone()`

The objects of this class are cloneable with this method.

#### Usage

    Tenmat$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
t <- tensor(array(1:24, dim = c(3, 4, 2)))
# Unfold mode 1 to rows, modes 2 and 3 to columns
m <- tenmat(t, rdims = 1)
print(m)
#> <Tenmat object>
#> A matrix corresponding to a tensor of size 3 x 4 x 2 
#> rindices = [ 1 ] (modes of tensor corresponding to rows)
#> cindices = [ 2 3 ] (modes of tensor corresponding to columns)
#> data =
#>      [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8]
#> [1,]    1    4    7   10   13   16   19   22
#> [2,]    2    5    8   11   14   17   20   23
#> [3,]    3    6    9   12   15   18   21   24

# Unfold mode 2 to rows, and mode 3 and 1 to columns (forward cyclic)
m2 <- tenmat(t, rdims = 2, cdims = "fc")
print(m2)
#> <Tenmat object>
#> A matrix corresponding to a tensor of size 3 x 4 x 2 
#> rindices = [ 2 ] (modes of tensor corresponding to rows)
#> cindices = [ 3 1 ] (modes of tensor corresponding to columns)
#> data =
#>      [,1] [,2] [,3] [,4] [,5] [,6]
#> [1,]    1   13    2   14    3   15
#> [2,]    4   16    5   17    6   18
#> [3,]    7   19    8   20    9   21
#> [4,]   10   22   11   23   12   24
```
