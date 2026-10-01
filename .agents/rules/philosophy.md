---
trigger: always_on
---

Look at the architecture and lession files in /doc folder

# Tensory: Architecture and Design Philosophy

## Overview

`tensory` is a lightweight, modern R package that brings high-performance tensor operations to R. The package uniquely pairs a user-friendly R6 class system on the frontend with a high-performance C++ (Rcpp) and BLAS backend.

## Design Philosophy

### 1. R6 Object-Oriented Frontend

By using the R6 class `Tensor`, the package provides clean encapsulation of tensor state (`data`, `dims`). It allows for both method-chaining (e.g., `t$clone_tensor()$add(t2)`) and native S3 operator overloading (`t + t2`), ensuring an intuitive interface.

### 2. Zero-Copy Memory Integration

Compiled kernels take tensor storage as `Rcpp::NumericVector`, which aliases R's own array buffer (`SEXP`) without copying; shapes come from the `dim` attribute via the helpers in `src/tensor_array.h`. Results are allocated as R arrays and written in place.

### 3. Optimized Linear Algebra

Instead of relying on naive loops or standard C++ algorithms for complex tensor contractions like Tensor Times Matrix (`ttm`), the package directly calls Fortran-based BLAS (`dgemm`) from C++. The design optimizes memory layouts and utilizes transpose permutations carefully to maintain cache efficiency and maximum computational throughput.

### 4. MATLAB Tensor Toolbox Compatibility

The API and behavior of functions like `ttm` closely mirror the popular MATLAB Tensor Toolbox. This maintains familiar semantics for users coming from MATLAB or engineering backgrounds, minimizing the learning curve. Always check the manual of MATLAB Tensor Toolbox if necessary.

Download MATLAB tensortool box to /dev folder from https://gitlab.com/tensors/tensor_toolbox/-/releases/v3.8 if they do not exist for your reference.

### 5. Automatic Dimension Management

Singleton dimensions are automatically managed and squeezed out during vector operations or partial contractions. This keeps tensor shapes clean and logical without requiring manual user intervention.

### 6. Speed and benchmark

The RCpp implementation must be faster than the existing R package rTensor from https://cran.r-project.org/web/packages/rTensor/index.html. Do externsive benchmark to find the best cpp implementation using optimized linear algebra strategy. You can use Fortran-based BLAS directly.

### &. Atomic design

Put the CPP implementation for each operation in a dedicated file.

## Codebase Structure

- **`R/tensor_class.R`**: Contains the frontend definition of the `Tensor` R6 class, handling basic element-wise arithmetic locally via R's fast internalized C routines.
- **`R/tensor_operations.R`**: Contains complex operations like `ttm` that act as thin wrappers around the compiled C++ code.
- **`src/tensor_ttm.cpp`**: Hand-optimized C++ tensor-times-matrix contraction, computing strides by hand to pipe memory chunks to `dgemm`.

## Recommended Refactoring / TODOs

- **Migrate Squeeze**: The `squeeze` behavior is currently enacted in R within the R6 class. As highlighted in the repository `README.md`, migrating this to the compiled backend is a possible improvement for better architectural consistency.
