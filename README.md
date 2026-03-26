# tensory

`tensory` is an R package for tensor algebra with a Tensor Toolbox-style interface. It provides dense tensors, matricized tensors, and decomposed tensor representations, with an R API that can use optimized C++ backends where available.

## Features
- Dense `Tensor` objects for multidimensional arrays
- `Tenmat`, `KTensor`, and `TTensor` representations for unfolded and decomposed tensors
- Tensor Toolbox-style operations such as `ttm()`, `ttt()`, `ttv()`, `mttkrp()`, `contract()`, `permute()`, and `symmetrize()`
- Rcpp/xtensor-backed implementation hooks for performance-critical kernels

## Installation

You can install the development version from source:

```r
# Clone the repository
# In R:
install.packages("devtools")
devtools::install_local("path/to/tensory")
```

## Usage

```r
library(tensory)

# Dense tensor creation
x <- tensor(array(1:24, dim = c(2, 3, 4)))

# Tensor-times-matrix and tensor-times-vector operations
A <- matrix(runif(6), nrow = 2)
y <- ttm(x, A, mode = 2)
z <- ttv(x, c(1, 2), mode = 1)

# Tensor reshaping helpers
x_perm <- permute(x, c(3, 1, 2))
x_vec <- vec(x)
```

## Project Structure
- `R/` — R interface and exported functions
- `src/` — C++ source code, using xtensor for core computations
- `inst/include/` — C++ headers for xtensor and related libraries
- `man/` — R documentation
- `tests/` — Unit tests

## Dependencies
- [xtensor](https://github.com/xtensor-stack/xtensor) (C++)
- [xtensor-r](https://github.com/xtensor-stack/xtensor-r) (C++)
- [Rcpp](https://cran.r-project.org/package=Rcpp)

## Current Priorities

- Stabilize shape semantics and keep scalar behavior consistent across dense operations
- Complete package documentation for the full exported API surface
- Move more dense tensor kernels, including reshape/squeeze-adjacent operations, into the xtensor-backed path where it improves performance

## License
This package is licensed under MIT. The repository also vendors third-party
headers under `inst/include/` that are distributed under BSD-style licenses.
See [LICENSE](LICENSE), [LICENSE.md](LICENSE.md), and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for details.

## Acknowledgements
- [xtensor](https://github.com/xtensor-stack/xtensor) and [xtensor-r](https://github.com/xtensor-stack/xtensor-r) projects
- Rcpp authors
