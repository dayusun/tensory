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

The C++ kernels require a C++20 compiler. `src/Makevars` sets `CXX_STD = CXX20` plus `-funroll-loops` (optimization level comes from R's default flags, typically `-O2`). Note `pkgbuild::compile_dll()` defaults to a `-O0` debug build — pass `debug = FALSE` before benchmarking.

## Architecture

### Frontend: R6 object, S3 dispatch

Public dense object is the R6 class `Tensor` (`R/tensor_class.R`) holding `$data` (R array) and `$dims` (integer vector). Two call styles coexist:

- method-style on the R6 object: `x$clone_tensor()$add(y)`
- S3 operator/function dispatch: `x + y`, `ttm(x, A, mode = 2)`, `permute(x, c(3, 1, 2))`

`Tensor$new(..., fast = TRUE)` is an internal fast-path constructor that skips validation. Use it from kernels that already know shapes are correct; never expose it through public API.

The package also exports `Tenmat` (matricization), `KTensor` (Kruskal/CP), `TTensor` (Tucker), `Sptensor` (sparse, subscript/value storage) + `Sptenmat`, `SymTensor` (compact symmetric storage), `SymKTensor` (symmetric CP), and `SumTensor` (lazy sum of parts), each with their own R6 class file and an `as.tensor()` method that materializes back to a dense `Tensor`. Conversion goes through `as.tensor.<class>()` S3 methods declared in `NAMESPACE`.

`ktensor()`/`ttensor()`/`sptensor()`/`sumtensor()` prepend their own class **before** `"Tensor"` in the S3 class vector. This is deliberate: specialized methods (e.g. `fnorm.KTensor`, `mttkrp.Sptensor`) win dispatch, while the `"Tensor"` operators serve as the shared entry point for mixed arithmetic — `+.Tensor` etc. branch on `Sptensor`/`SumTensor` operands (sparse-preserving fast paths live in `.sptensor_arith`; `SumTensor` appends parts). Never give these classes their own S3 arithmetic methods: two *different* methods on the two operands of a binary op is an "incompatible methods" error in R, which is exactly what the shared-`Tensor`-method design avoids.

Structured (non-densifying) implementations exist for `fnorm`/`innerprod`/`nvecs`/`permute` on `KTensor` and `TTensor`, and for `fnorm`/`innerprod`/`mttkrp`/`nvecs`/`ttv`/`ttm`/`collapse`/`permute` on `Sptensor` (`.ttm_sparse` in `R/sptensor_methods.R` is dispatched from the `ttm()` generic). `cp_als` accepts an `Sptensor` natively and never densifies. When adding a new reduction, add the structured method rather than relying on the dense fallback.

`DESCRIPTION` has a `Collate:` field — class files load before method files, and `tensor_operations.R` depends on the class definitions. If you add a new R file, append it to `Collate` (or rerun `devtools::document()` which will not reorder the existing list).

### R / C++ split and the fallback-dispatch pattern

`doc/architecture.md` is the authoritative design doc. The split:

- Compiled kernels live in three `.cpp` files: `src/tensor_ttm.cpp` (`ttm_cpp`, `ttm_multiple_cpp`), `src/tensor_dense.cpp` (`mttkrp_cpp`, `mttkrps_cpp`, `fibers_cpp`, `contract_cpp`, `mask_cpp`, `issymmetric_cpp`), and `src/tensor_decomposition.cpp` (`khatri_rao_pair_cpp`, `mttkrp_blas_cpp`). Every kernel takes tensor storage as `xt::rarray<double>` wrapping R's SEXP.
- **Every C++ kernel is optional.** The R method keeps a full pure-R implementation and delegates only when the compiled symbol is present, guarded by `if (exists("<fn>_cpp", mode = "function")) return(<fn>_cpp(...))` before the R fallback (see `R/tensor_dense_methods.R`, `R/tensor_ttm.R`, `R/tensor_operations.R`). When editing either side, keep the two paths behaviorally identical — tests run against whichever is compiled. `mttkrp` tries `mttkrp_blas_cpp` first, then `mttkrp_cpp`, then R.
- `ttt` (tensor-times-tensor) is deliberately R-only: it uses `reshape` + `permute` + `%*%`. A prior C++ `xt::linalg::tensordot` prototype was up to 18× slower than R's native `aperm` + BLAS path, so the R implementation is the correct one. Do not rewrite `ttt` in C++ without benchmarks proving a win on representative shapes — see `doc/lesson.md`.
- `nvecs`, `symmetrize`, and the arithmetic/operator methods remain R-only in `R/tensor_dense_methods.R` / `R/tensor_operations.R` — future C++ candidates only *if* benchmarks justify it; do not move them speculatively.

### Decomposition layer

`cp_als` (`R/cp_decomposition.R`), `tucker_als` + `hosvd` (`R/tucker_decomposition.R`) are R-level ALS/HOSVD drivers that call the `mttkrp`/`ttm`/`nvecs` methods (so they inherit the C++ acceleration transparently). They return `KTensor` / `TTensor` objects. `cp_als` uses `.pinv` (SVD-based pseudoinverse) and `.fixsigns_cp` for sign convention; keep the Khatri–Rao product order consistent with `mttkrp`'s mode convention when touching these.

Further algorithms (all R-level, all riding on the accelerated primitives):

- `R/cp_variants.R` — `cp_nmu` (nonneg multiplicative updates), `cp_apr` (Poisson CP, Chi–Kolda MU), `cp_opt`/`cp_wopt` (L-BFGS-B on the exact gradient; `cp_wopt` handles missing data via a weight tensor), `cp_arls` (uniformly sampled ALS using `fibers()`; deliberately no FFT mixing — documented divergence from MATLAB). Shared helpers: `.kr_others` (skip-mode Khatri–Rao whose row order matches `unfold(x, rdims = n)` columns — tested against `mttkrp`), `.cp_fit`, `.factors_to_vec`/`.vec_to_factors`.
- `R/cp_sym.R` — `cp_sym` (symmetric CP via exact `ttsv`-based gradients, returns `SymKTensor`), `tucker_sym` (shared-subspace HOOI).
- `R/gcp_opt.R` — `gcp_opt` with a loss catalog (gaussian, poisson, poisson-log, bernoulli-odds/logit, rayleigh, gamma, huber, or custom `list(f, g, lower)`).
- `R/tensor_eigen.R` — `eig_sshopm` (adaptive SS-HOPM) and `eig_geap` (generalized eigenpairs), implemented from Kolda & Mayo (2014) Algorithms 1–2 including the exact eq. (3.3) Hessian; `ttsv(A, x, -2)` supplies `A x^{m-2}`. The Kofidis–Regalia benchmark in `test-eigen.R` pins the known eigenvalues.
- `R/tensor_constructors.R` — `tenrand`, `teneye` (built as `symmetrize` of a delta-product tensor; property-tested via `ttsv`), `tendiag`, and `export_data`/`import_data` (MATLAB Tensor Toolbox text format).
- KTensor post-fit utilities in `R/ktensor_methods.R`: `arrange`, `normalize`, `fixsigns`, `score` (greedy factor-match), `ncomponents`, `extract`, `redistribute`, `tovec`, `viz`.
- `R/tepls.R` — `tepls()`, the one **supervised** method (tensor predictor + response), from Zhang & Li (2017) Tensor Envelope PLS. It is *not* Tensor Toolbox; it returns a `tepls` S3 object with a `predict.tepls`/`coef`/`print`. Algorithm 4 (per-mode SIMPLS envelope bases) is implemented exactly; the mode-`k` second-moment matrix was cross-checked against `TEReg::TensPLS_fit`'s `U U^T`. Two documented divergences: TEReg uses an `EnvMU` envelope optimizer for the bases (we use the paper's SIMPLS deflation), and the reduced regression is fit as plain latent-score OLS (Algorithm 4 Step 6), which is invariant to the separable-covariance scale ambiguity. Predictor input is either a list of same-shaped observations or an order-`(m+1)` tensor with observations in the **last** mode.

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
