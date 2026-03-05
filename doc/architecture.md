# Tensory: Architecture and Design Philosophy

## Overview

`tensory` is a lightweight, modern R package that brings high-performance tensor operations to R. The package uniquely pairs a user-friendly R6 class system on the frontend with a high-performance C++ `xtensor` and BLAS backend.

## Design Philosophy

### 1. R6 Object-Oriented Frontend

By using the R6 class `Tensor`, the package provides clean encapsulation of tensor state (`data`, `dims`). It allows for both method-chaining (e.g., `t$clone_tensor()$add(t2)`) and native S3 operator overloading (`t + t2`), ensuring an intuitive interface.

### 2. Zero-Copy Memory Integration

The package utilizes `xt::rarray` from the `xtensor-r` library. This allows the C++ backend to directly map R arrays and matrices (`SEXP`) into the C++ `xtensor` ecosystem without any expensive memory copying.

### 3. Optimized Linear Algebra

Instead of relying on naive loops or standard C++ algorithms for complex tensor contractions like Tensor Times Matrix (`ttm`), the package directly interfaces with Fortran-based BLAS (`dgemm`) via `xtensor-blas`. The design optimizes memory layouts and utilizes transpose permutations carefully to maintain cache efficiency and maximum computational throughput.

### 4. MATLAB Tensor Toolbox Compatibility

The API and behavior of functions like `ttm` and `ttt` closely mirror the popular MATLAB Tensor Toolbox. This maintains familiar semantics for users coming from MATLAB or engineering backgrounds, minimizing the learning curve. For instance, full inner-contractions using `ttt` evaluate directly to `double` numeric scalars instantly, matching scalar mathematics perfectly while avoiding 0-dimension Object allocation overhead.

### 5. Synergy Between Native R and C++ memory

A core architectural pillar is knowing _when_ to use C++ versus R:

1. **Native R**: For complex multi-mode dimensional contractions (like `ttt`), `tensory` relies strictly on native R wrappers querying `.Internal(aperm)` and utilizing base `%*%`. R's multi-threaded C core logic heavily outperforms generalized C++ multidimensional iterator templates (like `xt::linalg::tensordot`) when memory layouts require complex mappings safely handled by hardware BLAS.
2. **Handcrafted C++**: For strict dimension slicing mappings (like `ttm`), `tensory` utilizes `xtensor` solely as a safe bridging and N-dimensional pointer query mechanism, while manually writing slicing loops over the data structs directly into subset `cblas_dgemm` calls. This drastically outperforms R's native behaviors by avoiding matrix transpositions entirely.

### 6. Automatic Dimension Management

Singleton dimensions are automatically managed and squeezed out during vector operations or partial contractions. This keeps tensor shapes clean and logical without requiring manual user intervention.

## Codebase Structure

- **`R/tensor_class.R`**: Contains the frontend definition of the `Tensor` R6 class, handling basic element-wise arithmetic locally via R's fast internalized C routines.
- **`R/tensor_operations.R`**: Contains complex operations like `ttm` that act as thin wrappers around the compiled C++ code.
- **`src/tensor_ttm.cpp` & `src/tensor_test.cpp`**: Hand-optimized C++ implementations of tensor times matrix/vector contractions, manipulating strides and `xtensor` views to pipe optimal memory chunks to `dgemm`.

## Recommended Refactoring / TODOs

- **Migrate Squeeze**: The `squeeze` behavior is currently enacted in R within the R6 class. As highlighted in the repository `README.md`, migrating this to the `xtensor` backend is a planned improvement for better architectural consistency.
