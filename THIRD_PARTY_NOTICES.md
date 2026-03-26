# Third-Party Notices

This repository includes vendored third-party C++ headers under `inst/include/`.
Those components are not covered by the repository's MIT license alone and
remain subject to their own licenses.

Included third-party components identified in this checkout:

- `inst/include/xtensor/`: BSD 3-Clause
- `inst/include/xtensor-r/`: BSD 3-Clause
- `inst/include/xtensor-blas/`: BSD 3-Clause
- `inst/include/xtl/`: BSD 3-Clause
- `inst/include/xsimd/`: BSD 3-Clause
- `inst/include/xflens/cxxblas/`: BSD-style license; see `inst/include/xflens/cxxblas/LICENSE`
- `inst/include/xflens/cxxlapack/`: BSD-style license; see `inst/include/xflens/cxxlapack/LICENSE`

Alignment notes:

- The `xtensor` family headers in this repository contain file-level notices
  stating they are distributed under the BSD 3-Clause License.
- The vendored `cxxblas` and `cxxlapack` directories already include local
  `LICENSE` files and should be preserved when redistributing this package.
- When redistributing source or binary builds that include these vendored files,
  keep the original copyright notices, license conditions, and disclaimers.

Official upstream license locations for the BSD 3-Clause components:

- `xtensor`: <https://github.com/xtensor-stack/xtensor/blob/master/LICENSE>
- `xtensor-r`: <https://github.com/xtensor-stack/xtensor-r/blob/master/LICENSE>
- `xtensor-blas`: <https://github.com/xtensor-stack/xtensor-blas/blob/master/LICENSE>
- `xtl`: <https://github.com/xtensor-stack/xtl/blob/master/LICENSE>
- `xsimd`: <https://github.com/xtensor-stack/xsimd/blob/master/LICENSE>
