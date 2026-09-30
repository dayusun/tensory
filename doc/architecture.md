# Tensory: Architecture and Design Philosophy

## Overview

`tensory` is an R package for tensor algebra that combines:

- an R6 and S3 frontend for user-facing tensor objects and operators
- compiled C++ kernels (plain Rcpp, no tensor library) for heavy dense contractions
- BLAS-backed matrix multiplications for performance-critical paths

The design goal is not to move every tensor operation into C++. The package is built around a split architecture: keep dispatch, shape management, and compatibility helpers simple in R, while pushing fixed-layout dense kernels into compiled code.

## Current Architecture

### 1. Frontend Model

The main public dense object is the R6 `Tensor` class. It encapsulates:

- `data`: the underlying R array
- `dims`: the tensor dimensions

This frontend supports both:

- method-style operations such as `x$clone_tensor()$add(y)`
- S3 operator overloading such as `x + y`, `x * y`, and comparisons

The package also exposes `Tenmat`, `KTensor`, and `TTensor` objects, with dense coercion paths through `as.tensor()`.

### 2. Memory and Tensor Representation

Compiled kernels take tensor storage as `Rcpp::NumericVector`, which aliases R's own column-major `REALSXP` buffer without copying (integer and logical arrays are converted to double once, as `as.double()` would). The shape is the array's `dim` attribute. `src/tensor_array.h` holds the few shared helpers: `array_dims()` (dim attribute, or length for a plain vector), `column_major_strides()`, `product_of_dims()`, and `alloc_array()`, which allocates an uninitialised R array with a `dim` attribute so a kernel can write its result directly into the object returned to R.

Earlier versions wrapped the storage in `xt::rarray` from `xtensor-r`. The kernels only ever used its data pointer and shape, while xtensor itself required C++20, headers fetched at build time, and an extra result copy in `ttm`, so it was dropped with no change in results or speed.

Reading input is zero-copy. `ttm` is zero-copy end to end except for tensors whose leading extent before the mode is 2 or 3, where it gathers into a buffer because per-slice `dgemm` calls that small are slower. `mttkrp` reads the tensor in place too: a two-step contraction only allocates Khatri-Rao products of one side's factors and an `R`-column partial result, never the unfolding.

### 3. Current R vs C++ Split

The current codebase uses a pragmatic split:

- **R** handles object construction, S3 dispatch, shape validation, simple reductions, compatibility wrappers, and dense helper methods.
- **C++ (Rcpp + BLAS)** is used where a dense kernel benefits from explicit control over tensor layout, shape metadata, and BLAS invocation.

At the moment:

- `ttm` uses a compiled C++ backend in `src/tensor_ttm.cpp`
- `ttt` is implemented in R using reshape, permute, and `%*%`
- the two hot spots of `spgtr()` are compiled in `src/tensor_spgtr.cpp`: mode-wise
  marginal covariances (a `dsyrk` accumulation per observation, replacing `m` array
  permutations) and the Stiefel-manifold proximal-gradient solver (the whole iteration
  runs in C++ on `dgemm`/`dsyev` rather than returning to R per step). Both keep the
  reference R implementations as fallbacks, pinned against the kernels in the tests.
- many dense helper functions in `R/tensor_dense_methods.R` are still R-level implementations

This split is intentional. The package does not assume that a generic C++ tensor expression is always faster than R reshaping plus BLAS.

### 4. Linear Algebra Backend

The core dense contraction path in `ttm` uses direct BLAS `dgemm` calls from C++ on R-owned storage: the tensor is viewed as `M1 x Ik x M2`, and each contiguous `M1 x Ik` slice is multiplied in place (one call for mode 1 and for the last mode), so the tensor is never permuted. The implementation is optimized around:

- predictable memory layout
- explicit stride computation
- one large BLAS call where possible
- avoiding repeated high-level tensor permutations in the hot loop

Performance therefore comes from the BLAS R is linked against (OpenBLAS, MKL or Accelerate rather than reference BLAS) and from how much memory the kernel moves around each BLAS call, not from a C++ tensor library.

### 5. MATLAB Tensor Toolbox Compatibility

The public API intentionally mirrors the MATLAB Tensor Toolbox where that improves usability for tensor users:

- `ttm`, `ttv`, `ttt`
- `tenmat`, `ktensor`, `ttensor`
- dense helpers such as `mttkrp`, `contract`, `mask`, `nvecs`, `symmetrize`, and `ttsv`

Compatibility is semantic, not literal. The package remains somewhat R-leaning in places where strict MATLAB behavior would be awkward or low-value in R.

One important example: full contractions in `ttt` currently return a **scalar `Tensor`** with `integer(0)` dimensions, not a bare R numeric scalar.

### 6. Dimension Management

Dense tensor operations manage dimensions explicitly:

- vector contractions reduce order
- singleton dimensions may be removed where that matches tensor algebra expectations
- scalar results are represented as scalar tensors

This keeps tensor shape behavior explicit and consistent with the package's object model, even when the underlying operation reduces to a single number.

## Current Codebase Structure

- `R/tensor_class.R`: `Tensor` R6 class and tensor operator methods
- `R/tensor_ttm.R`: dense tensor-times-matrix/vector dispatch and wrappers
- `R/tensor_ttt.R`: dense tensor-times-tensor contractions in R
- `R/tensor_dense_methods.R`: dense tensor helper methods and compatibility functions
- `R/tepls.R`, `R/spgtr.R`: supervised tensor-predictor regression (continuous and GLM)
- `src/tensor_array.h`: shared shape/stride helpers for the compiled kernels
- `src/tensor_ttm.cpp`: compiled `ttm` kernel using explicit layout handling and BLAS
- `src/tensor_decomposition.cpp`: two-step `mttkrp` / `mttkrps` kernels and the Khatri-Rao kernel
- `src/tensor_dense.cpp`: elementwise `mttkrp`, `fibers`, `contract`, `mask`, `issymmetric`
- `bench/kernels.R`, `bench/compare.R`: kernel benchmark harness and build-to-build comparison
- `src/tensor_spgtr.cpp`: compiled mode-covariance and manifold-solver kernels for `spgtr`

## Target Architecture

### 1. Stable Principle

The target architecture is:

- **R for orchestration**
- **Rcpp + BLAS C++ for dense kernels**

That means:

- keep public object semantics, dispatch, and lightweight wrappers in R
- move repeated heavy dense kernels into compiled code when they are layout-sensitive or benchmark-significant
- share shape, stride and allocation helpers through `src/tensor_array.h` rather than reimplementing them per file

### 2. Dense Hotspots to Compile

The main dense performance target after `ttm` is to move more of these kernels into compiled code:

- `mttkrp`
- `mttkrps`
- `fibers`
- `contract`
- `mask`
- symmetry checks used by `issymmetric` and `symmetrize`

These functions all depend on the same small set of low-level operations:

- column-major stride calculation
- subscript-to-linear-index conversion
- controlled mode reordering
- BLAS-friendly contraction layouts

Those helpers should be shared rather than reimplemented piecemeal in R. The preferred implementation style is:

- R-owned storage accessed through `Rcpp::NumericVector`, with results allocated as R arrays and written in place
- direct BLAS for the matrix multiply core when the kernel reduces cleanly to GEMM-like work
- small handcrafted layout helpers, preferring strided BLAS calls on slices over gathering into temporary buffers

### 3. R-Level Functions That Can Stay in R

Not every dense function needs a dedicated C++ kernel. Functions can remain primarily R-level when they are:

- thin compatibility wrappers
- simple aliases
- dominated by existing optimized base R operations
- not meaningful hotspots in benchmarks

Examples include:

- `tenfun`
- `isequal`
- `isscalar`
- `full`
- `double`
- `scale` as a wrapper around tensor scaling logic

### 4. Documentation Rule

The architecture documentation should always distinguish between:

- **what is true in the current code**
- **what is the target direction**

Ambitious claims are acceptable only when labeled as future-state goals. Current-state sections should describe the implementation as it actually exists in the repository.

## Design Principles

### 1. Prefer Correct Layout Control Over Generic Tensor Expressions

For dense hotspots, explicit layout control is preferred over generic tensor-expression abstractions when it improves correctness, predictability, or BLAS utilization. A `xt::linalg::tensordot` prototype of `ttt` was up to 18x slower than R's `aperm` + BLAS (`doc/lesson.md`); a C++ tensor library is worth adding only for a specific kernel where benchmarks show it beats BLAS on R's own storage.

### 2. Keep the Public API Familiar

Tensor Toolbox naming and semantics are worth preserving for core tensor algebra, because they lower the barrier for users coming from MATLAB, engineering, and scientific computing workflows.

### 3. Stay R-Leaning Where It Helps

Compatibility does not require copying every MATLAB object behavior exactly. When there is no algebraic reason to mimic MATLAB literally, the package should prefer the clearer R-facing behavior.

### 4. Benchmark Before Moving More Code to C++

Compiled code is not automatically better. Additional kernels should move into C++ only when:

- they are on a hot path
- they benefit from layout-aware implementation
- benchmarks show that the compiled path wins meaningfully

Use `bench/kernels.R` to time a change and `bench/compare.R` to compare builds; `compare.R` also fails if two builds' results differ.

## Near-Term TODOs

- Move additional dense hotspots beyond `ttm` to compiled kernels, starting with `mttkrp`
- Consolidate shared dense index and stride helpers in C++
- Benchmark compiled and R-level dense kernels on representative tensor sizes (`bench/kernels.R`; extend its cases as kernels change)
- Revisit whether `squeeze` should remain in R or move to compiled code based on profiling rather than architectural purity alone
