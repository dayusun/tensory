# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common commands

R package built with roxygen2 + Rcpp + testthat 3.

- Load for interactive use: `R -e 'devtools::load_all(".")'`
- Regenerate Rd + NAMESPACE after roxygen edits: `R -e 'devtools::document()'`
- Regenerate `R/RcppExports.R` and `src/RcppExports.cpp` after editing `// [[Rcpp::export]]` signatures: `R -e 'Rcpp::compileAttributes()'` (must run before `load_all`/`document` if C++ exports changed)
- Run all tests: `R -e 'devtools::test()'`
- Run a single test file: `R -e 'devtools::test(filter = "ttm")'` (matches `tests/testthat/test-ttm.R`)
- Full check (run before tagging a release): `R -e 'devtools::check()'`
- Build vignettes: `R -e 'devtools::build_vignettes()'`
- Re-vendor xtensor / xtl / xsimd / xtensor-blas headers into `inst/include/` from upstream tags: `bash inst/tools/vendor` (only needed when bumping pinned versions inside that script; xtensor-r is intentionally not re-fetched there)

The C++ kernels require a C++20 compiler. `src/Makevars` sets `CXX_STD = CXX20` plus `-O3 -march=native`; if a contributor's toolchain rejects `-march=native`, edit `Makevars` rather than overriding globally.

## Architecture

### Frontend: R6 object, S3 dispatch

Public dense object is the R6 class `Tensor` (`R/tensor_class.R`) holding `$data` (R array) and `$dims` (integer vector). Two call styles coexist:

- method-style on the R6 object: `x$clone_tensor()$add(y)`
- S3 operator/function dispatch: `x + y`, `ttm(x, A, mode = 2)`, `permute(x, c(3, 1, 2))`

`Tensor$new(..., fast = TRUE)` is an internal fast-path constructor that skips validation. Use it from kernels that already know shapes are correct; never expose it through public API.

The package also exports `Tenmat` (matricization), `KTensor` (Kruskal/CP), and `TTensor` (Tucker), each with their own R6 class file and an `as.tensor()` method that materializes back to a dense `Tensor`. Conversion goes through `as.tensor.<class>()` S3 methods declared in `NAMESPACE`.

`DESCRIPTION` has a `Collate:` field — class files load before method files, and `tensor_operations.R` depends on the class definitions. If you add a new R file, append it to `Collate` (or rerun `devtools::document()` which will not reorder the existing list).

### R / C++ split is intentional and asymmetric

`doc/architecture.md` is the authoritative design doc. The current split:

- `ttm` (tensor-times-matrix) is the only operation with a compiled kernel — `src/tensor_ttm.cpp`, dispatched from `R/tensor_ttm.R`.
- `ttt` (tensor-times-tensor) is deliberately R-level: it uses `reshape` + `permute` + `%*%`. A prior C++ `xt::linalg::tensordot` prototype was up to 18× slower than R's native `aperm` + BLAS path, so the R implementation is the correct one. Do not rewrite `ttt` in C++ without benchmarks proving a win on representative shapes — see `doc/lesson.md` for the full reasoning.
- All other dense helpers (`mttkrp`, `contract`, `mask`, `nvecs`, `symmetrize`, `fibers`, etc.) currently live in `R/tensor_dense_methods.R`. The architecture doc identifies these as future C++ candidates *if* benchmarks justify it; do not move them speculatively.

### `ttm` C++ kernel — current vs. target

`src/tensor_ttm.cpp` currently does extract → single `dgemm` → scatter:

1. Gather every contracted-mode fiber into a contiguous `Ik × rest` buffer (`x_mat`).
2. One `dgemm` call producing the `J × rest` result buffer (`y_mat`).
3. Scatter back into the result tensor with the contracted dim in its original position.

This is **not** zero-copy — there are two intermediate copies plus the `xt::xarray` → `xt::rarray` allocation. Before adding optimizations, benchmark first; the current single-shot dgemm is straightforward and correct.

`doc/lesson.md` describes the *target* design (per-`M2` slice loop, mode-1 single-shot fast path, `M2 == 1 && M1 > 2000` cache-tiled branch with 512-row blocks). That target is not yet in the code — treat lesson.md as design intent, not implementation reference. If you implement the slice loop, preserve the lesson's three-branch structure (mode-1 fast path, middle-mode loop, mode-N cache-tiled).

`dgemm` is invoked with `transa = "T"|"N"` driven by `transpose`, `transb = "N"`, leading dims `lda = transpose ? Ik : J`, `ldb = Ik`, `ldc = J`. There is an explicit `INT_MAX` guard before casting `size_t` extents to the BLAS `int` parameters — keep it when refactoring.

### xtensor integration

`inst/include/` vendors `xtensor`, `xtl`, `xsimd`, `xtensor-r`, `xtensor-blas`, `xflens`. `inst/include/tensory.h` exposes only `xtensor-r/rarray.hpp` and `roptional.hpp` — the C++ side uses `xt::rarray` to wrap R-owned SEXP storage without copying, then drops to raw `dgemm` for the contraction core. This is **handcrafted xtensor + BLAS**, not the high-level `xt::linalg::tensordot` path.

`src/Makevars` links `$(LAPACK_LIBS) $(BLAS_LIBS) $(FLIBS)` so the kernel uses whatever BLAS R was built against.

### Scalar tensor convention

A full contraction (e.g. `ttt` with all modes contracted, or `ttv` reducing a 1-D tensor) returns a **scalar `Tensor` with `dims = integer(0)`**, not a bare R numeric. Operations and tests rely on this — when adding new reductions, return `Tensor$new(value, integer(0), fast = TRUE)` rather than unwrapping to a numeric. `isscalar()` is the predicate that recognizes this shape.

### `ttm` automatic squeeze

`ttm` automatically squeezes singleton dimensions introduced by vector multiplications (a vector contraction collapses its mode). When passed a list of matrices/vectors via `ttm(x, list(...), mode = c(...))`, contractions are sorted by mode in descending order and a single final permute restores axis ordering — do not add per-step transposes.

### MATLAB Tensor Toolbox API parity

Naming and argument semantics intentionally track the MATLAB Tensor Toolbox (`ttm`, `ttv`, `ttt`, `tenmat`, `ktensor`, `ttensor`, `mttkrp`, `nvecs`, `symmetrize`, `ttsv`). Parity is *semantic*, not literal — when MATLAB behavior would be awkward in R (e.g., scalar return shapes), prefer the R-leaning choice and document the divergence. The repository does **not** vendor Tensor Toolbox source; if any MATLAB code is ever copied in, its BSD-2-Clause notice must be retained (see `THIRD_PARTY_NOTICES.md`).
