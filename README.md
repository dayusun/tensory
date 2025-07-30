# tensory

`tensory` is an R package for general tensor operations, providing a fast and flexible interface for multidimensional array computations. It leverages the C++ [xtensor](https://github.com/xtensor-stack/xtensor) library to deliver high-performance numerical routines, similar to NumPy in Python, but accessible from R.

## Features
- General tensor (multidimensional array) operations in R
- High performance via C++ backend using xtensor
- Rcpp integration for seamless R/C++ interoperability
- Extensible and efficient, suitable for scientific computing and data analysis

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

# Example: Create and manipulate a tensor
# (See package documentation for available functions)
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

## TODO

### Planned Improvements
- [ ] **Convert squeeze method to xtensor implementation**: Currently the squeeze functionality in the Tensor class is implemented at the R level. Plan to migrate this to use xtensor's built-in squeeze function for better performance and consistency with the C++ backend architecture.

## License
See [LICENSE](LICENSE) for details.

## Acknowledgements
- [xtensor](https://github.com/xtensor-stack/xtensor) and [xtensor-r](https://github.com/xtensor-stack/xtensor-r) projects
- Rcpp authors
