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
- Benchmark the compiled kernels against an installed build: `R_LIBS=<lib> Rscript bench/kernels.R out.csv label`, then `Rscript bench/compare.R old.csv new.csv` (prints speedups and exits non-zero if the two builds' results differ). Install each build with `R CMD INSTALL --library=<lib> .` so it gets R's normal `-O2` flags.
- Large-tensor sweep of `ttm`/`mttkrp`/`mttkrps` over every mode of order-3 to order-6 tensors (~64M elements; needs ~5 GB RAM for the old kernels): `R_LIBS=<lib> Rscript bench/sweep.R out.csv label`, compared the same way with `bench/compare.R`.

The C++ kernels need only Rcpp and a C++11-or-later compiler; no headers are vendored or fetched. `src/Makevars` sets only `PKG_LIBS` — the language standard is R's default and optimization flags come from R's own `CXXFLAGS`/`CXX17FLAGS` (typically `-O2`), because `R CMD check` warns about any `-O`/`-f`/`-m` tuning set in `PKG_CXXFLAGS`. Put local tuning (e.g. `-funroll-loops`) in `~/.R/Makevars`. Note `pkgbuild::compile_dll()` defaults to a `-O0` debug build — pass `debug = FALSE` before benchmarking.

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

- Compiled kernels live in four `.cpp` files: `src/tensor_ttm.cpp` (`ttm_cpp`, `ttm_multiple_cpp`), `src/tensor_dense.cpp` (`mttkrp_cpp`, `fibers_cpp`, `contract_cpp`, `mask_cpp`, `issymmetric_cpp`), and `src/tensor_decomposition.cpp` (`khatri_rao_pair_cpp`, `mttkrp_blas_cpp`, `mttkrps_cpp`). `src/tensor_spgtr.cpp` holds the `spgtr` kernels. Every kernel takes tensor storage as `Rcpp::NumericVector`, which aliases R's buffer without copying; shared shape helpers are in `src/tensor_array.h`.
- **Every C++ kernel is optional.** The R method keeps a full pure-R implementation and delegates only when the compiled symbol is present, guarded by `if (exists("<fn>_cpp", mode = "function")) return(<fn>_cpp(...))` before the R fallback (see `R/tensor_dense_methods.R`, `R/tensor_ttm.R`, `R/tensor_operations.R`). When editing either side, keep the two paths behaviorally identical — tests run against whichever is compiled. `mttkrp` tries `mttkrp_blas_cpp` first, then `mttkrp_cpp`, then R; `mttkrps` tries `mttkrps_cpp`, then per-mode `mttkrp_blas_cpp`, then R.
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
- `R/tepls.R` — `tepls()`, the one **supervised** method (tensor predictor + response), from Zhang & Li (2017) Tensor Envelope PLS. It is *not* Tensor Toolbox; it returns a `tepls` S3 object with a `predict.tepls`/`coef`/`print`. Algorithm 4 (per-mode SIMPLS envelope bases) is implemented exactly; the mode-`k` second-moment matrix was cross-checked against `TEReg::TensPLS_fit`'s `U U^T`. Two documented divergences: TEReg uses an `EnvMU` envelope optimizer for the bases (we use the paper's SIMPLS deflation), and the reduced regression is fit as plain latent-score OLS (Algorithm 4 Step 6), which is invariant to the separable-covariance scale ambiguity. Predictor input is either a list of same-shaped observations or an order-`(m+1)` tensor with observations in the **last** mode. `u = NULL` (the default) delegates to `spgtr.R`'s `.spgtr_auto_u` eigenvalue-gap rule, and the mode covariances go through `.spgtr_mode_covs` so the compiled `spgtr_mode_covs_cpp` kernel applies here too — hence the centered design is kept as `Xt` (`prod(p) x n`), not `Xc`. Vignette: `vignettes/tepls.Rmd`.

- `R/spgtr.R` — `spgtr()`/`spgtr_cv()`, the GLM counterpart of `tepls()` (port of the MATLAB TPLSGLM/SPGTR code): the response enters through the working residual `y - mu_0` of the nuisance-only GLM, per-mode SIMPLS bases are optionally refined by minimizing the Cook–Zhang envelope objective over the Stiefel manifold, and an adaptive row-wise L2,1 penalty (`lambda > 0`) drops whole slices of the predictor. The manifold solver is SLPG (Xiao–Liu–Yuan) with a polar retraction: **every retraction/feasibility step must right-multiply `W`** — that is what preserves the row sparsity the prox step creates, so never swap in a QR retraction (MATLAB's `rank(X) < p` QR fallback silently refills zeroed rows; here a rank-deficient iterate halves the step and eventually stops). The fit is split into `.spgtr_prepare()` (covariances, signal matrices, SIMPLS bases — everything independent of `lambda`) and `.spgtr_solve()` (bases at one `lambda` + score-level GLM) so `spgtr_cv()` reuses one preparation per fold and warm-starts along the path. Shares `.tepls_design`/`.tepls_mode_covs`/`.tepls_simpls_mode` with `tepls.R`; `.sym_pow` lives here and `tepls.R`'s `.sym_inv` delegates to it. Not ported from MATLAB: the symmetric-predictor (`ifsym`) variant, the likelihood-based `gpls_tensor_lik_core`, and the elastic-net fallback used there when the score-level GLM separates. Vignette: `vignettes/spgtr.Rmd`.
- `R/pqtr.R` — `pqtr()`/`pqtr_cv()`, the **quantile** counterpart of `tepls()` (port of https://github.com/dayusun/PQTR, the MATLAB code for Sun, Qiu, Peng, Guo & Manatunga 2024, JASA). The response enters through the working residual `tau - 1{y < Q_tau(y | Z)}` — the subgradient of the check loss at the nuisance-only quantile fit — after which the per-mode signal matrices and the reduced fit are structurally the same as `spgtr`'s. Three shared helpers make that possible: `.pls_signal()` (in `spgtr.R`, also used by `.spgtr_prepare`), `.spgtr_auto_u`, and `.tepls_simpls_mode`. **`.tepls_simpls_mode` gained an `orth` argument**: `"scores"` (default, de Jong SIMPLS, `w_s' Sigma w_t = 0`) for `tepls`/`spgtr`, `"weights"` (oblique projector `I - Sigma W (W' Sigma W)^-1 W'`, leaving `W'W = I`) for `pqtr`, because that is what `pls_tensor_core.m` does. The two span the same subspace only when the signal matrix is rank one — `test-pqtr.R` pins both the difference and that coincidence. Quantile regressions are solved by `.rq_fit()`, an MM iteration (Hunter & Lange 2000) annealing `eps` from 1e-1 to 1e-8; deliberately no `quantreg` dependency. Documented divergences from MATLAB: MM instead of `fminunc`, predictor centered before scoring (changes `alpha`, not `B`), `pqtr_cv()` takes an explicit `u_grid` instead of enumerating `d^m` combinations, and the eigenvalue-ratio search is capped at 5 candidates per mode instead of `sqrt(n - q)` (the wide search reliably returns the rank-cliff ratio; `.spgtr_auto_u` also now drops numerically-zero eigenvalues before applying the rule). Vignette: `vignettes/pqtr.Rmd`.
- `src/tensor_spgtr.cpp` — `spgtr_mode_covs_cpp` (per-observation `dsyrk` accumulation of `Sigma_k`, no array permutation; three branches for `L == 1` / `R == 1` / gather) and `spgtr_slpg_cpp` (the whole manifold prox-gradient loop, raw `dgemm`/`dsyev`). Both follow the package's fallback-dispatch rule with `.tepls_mode_covs` and `.env_slpg_r` as the reference R paths — `test-spgtr.R` pins the two against each other, so any edit must touch both. Latent scores deliberately go through `ttm()` rather than a Kronecker product. Measured: the Kronecker path is ~2-3x *faster* for order-2 predictors with tiny `u` (milliseconds either way), but `ttm` wins 1.3x at `u = (3,3,3)` and 5.4x at `u = (5,5,5)` and never allocates the `prod(p) x prod(u)` factor — so `ttm` is the single path, at a few ms cost in the cheap case.

### `ttm` C++ kernel

`src/tensor_ttm.cpp` views the column-major tensor as `M1 × Ik × M2` (`M1` = product of dims before the mode, `M2` = after) and never permutes it:

1. `M1 == 1` (mode 1): the tensor is already an `Ik × M2` matrix — one `dgemm` on R's storage, `transa = transpose ? "T" : "N"`.
2. `M1 >= kGatherBelowM1` (4): one `dgemm` per `M2` slice, `Y_s = X_s A'` (`transb = transpose ? "N" : "T"`, `lda = ldc = M1`), written straight into the result. For the last mode `M2 == 1`, so this is a single call.
3. `1 < M1 < 4`: gather into an `Ik × (M1 M2)` buffer, one `dgemm`, scatter — per-slice calls that small lose to call overhead (2× at `M1 = 2` on reference BLAS; tie at 3; slicing wins from 4 on both reference BLAS and OpenBLAS).

`doc/lesson.md`'s 512-row cache tiling for `M2 == 1 && M1 > 2000` was benchmarked and **deliberately left out**: 0–13% faster on reference BLAS but 1.2–2.3× slower on OpenBLAS, which already blocks for cache inside `dgemm`. Re-benchmark on both BLAS libraries before revisiting it or `kGatherBelowM1`. Every extent goes through `blas_int()` (the `INT_MAX` guard) before being passed to BLAS — keep it when refactoring. `test-ttm.R` pins every branch against `.ttm_matrix_base`.

To compare BLAS libraries on Debian/Ubuntu, `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/<blas|openblas-pthread>/libblas.so.3` selects one per process (`LD_LIBRARY_PATH` is reset by R's `ldpaths`).

### `mttkrp` C++ kernels

Both are two-step MTTKRPs (Phan, Tichavský & Cichocki 2013) in `src/tensor_decomposition.cpp`; neither forms the mode-`n` unfolding nor the full `prod(I_k, k != n) × R` Khatri–Rao product. `kr_range(first, last)` builds the Khatri–Rao product of a consecutive block of modes, rows in column-major order with mode `first` fastest.

- `mttkrp_blas_cpp` (`mttkrp_two_step`) picks an "outer" block of modes at one end of the tensor — a suffix `[c, N)` with `c > n` or a prefix `[0, c)` with `c <= n` — minimizing `Mo + total / Mo`, so `Mo` lands near `sqrt(total)`. One `dgemm` contracts R's storage with that block's Khatri–Rao product, leaving an `a × In × b` partial per column; two `dgemv` calls per column finish it against the Khatri–Rao products of the modes left on each side of `n`. Those side products are always applied, even when a side has only singleton modes: their 1 × R rows are **not** all ones (dropping them was a real bug caught by `test-cp.R`'s singleton shapes). Taking the whole larger side as the outer block instead builds a `total / In`-row Khatri–Rao product for mode 1 and mode N, which dominated high-order runs.
- `mttkrps_cpp` (all modes, fixed factors) splits the modes into `[0, s)` / `[s, N)` with `max(M_L, M_R)` minimized, does one `dgemm` per side (two passes over the tensor instead of `N`), and `finish_group()` completes each mode from its side's `M_side × R` partial tensor.

`test-cp.R` pins both against the elementwise `mttkrp_cpp` across orders 2–5, ranks 1 and 3, and singleton modes. `cp_als` still calls `mttkrp()` once per mode because its factors change between modes; a dimension-tree ALS (reuse one side's partial across that side's updates) is the next step if `cp_als` needs more speed.

### C++ backend: Rcpp + BLAS, no tensor library

Kernels read R arrays through `Rcpp::NumericVector` (zero-copy; the shape is the `dim` attribute, via `tensory::array_dims`), allocate results with `tensory::alloc_array` and write into them in place, and call Fortran BLAS/LAPACK (`dgemm`, `dsyrk`, `dsyev`) directly. xtensor was used previously and removed: the kernels only used its data pointer and shape, while it forced C++20 and build-time header fetching. `inst/include/tensory.h` is included by `RcppExports.cpp` and pulls in only Rcpp. Do not reintroduce a C++ tensor library without a benchmark (`bench/`) showing it beats BLAS on R's storage for a specific kernel. Performance comes from the BLAS R links against and from how little memory a kernel moves around each BLAS call.

`src/Makevars` links `$(LAPACK_LIBS) $(BLAS_LIBS) $(FLIBS)` so the kernel uses whatever BLAS R was built against.

### Scalar tensor convention

A full contraction (e.g. `ttt` with all modes contracted, or `ttv` reducing a 1-D tensor) returns a **scalar `Tensor` with `dims = integer(0)`**, not a bare R numeric. Operations and tests rely on this — when adding new reductions, return `Tensor$new(value, integer(0), fast = TRUE)` rather than unwrapping to a numeric. `isscalar()` is the predicate that recognizes this shape.

### `ttm` automatic squeeze

`ttm` automatically squeezes singleton dimensions introduced by vector multiplications (a vector contraction collapses its mode). When passed a list of matrices/vectors via `ttm(x, list(...), mode = c(...))`, contractions are sorted by mode in descending order and a single final permute restores axis ordering — do not add per-step transposes.

### MATLAB Tensor Toolbox API parity

Naming and argument semantics intentionally track the MATLAB Tensor Toolbox (`ttm`, `ttv`, `ttt`, `tenmat`, `ktensor`, `ttensor`, `mttkrp`, `nvecs`, `symmetrize`, `ttsv`). Parity is *semantic*, not literal — when MATLAB behavior would be awkward in R (e.g., scalar return shapes), prefer the R-leaning choice and document the divergence. The repository does **not** vendor Tensor Toolbox source; if any MATLAB code is ever copied in, its BSD-2-Clause notice must be retained (see `THIRD_PARTY_NOTICES.md`).
