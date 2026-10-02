# Third-Party Notices

`tensory` does not vendor third-party source code. The compiled kernels use
only Rcpp (a `LinkingTo` dependency) and the BLAS/LAPACK libraries R itself
is linked against.

Earlier development versions vendored the BSD 3-Clause licensed `xtensor`,
`xtensor-r`, `xtensor-blas`, `xtl` and `xsimd` headers, and the BSD-style
`xflens` `cxxblas`/`cxxlapack` headers, under `inst/include/`. Those headers
and the script that fetched them were removed once no kernel used them; they
are not part of this package.

MATLAB Tensor Toolbox note:

- `tensory` aims for an API similar to the MATLAB Tensor Toolbox.
- The Tensor Toolbox for MATLAB is published under the BSD 2-Clause license.
- This repository does not currently vendor Tensor Toolbox source code.
- If any Tensor Toolbox source or documentation is incorporated in the future,
  its original copyright notice and BSD 2-Clause license terms must be
  preserved.
