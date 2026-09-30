#include "tensor_array.h"
#include <Rcpp.h>
#include <algorithm>
#include <climits>
#include <vector>


// Use R's BLAS interface via Fortran calls (standard R package approach)
// Fortran name mangling
#ifndef F77_NAME
#define F77_NAME(x) x##_
#endif

#ifndef FCONE
#define FCONE
#endif

// Externally declare the Fortran BLAS subroutines
extern "C" {
void F77_NAME(dgemm)(const char *transa, const char *transb, const int *m,
                     const int *n, const int *k, const double *alpha,
                     const double *a, const int *lda, const double *b,
                     const int *ldb, const double *beta, double *c,
                     const int *ldc FCONE FCONE);
}

using namespace Rcpp;
using tensory::column_major_strides;
using tensory::product_of_dims;

/**
 * @brief Tensor-times-matrix operation with optional transpose
 * @param tensor_data Input tensor as an R array (aliased, not copied)
 * @param matrix Input matrix
 * @param mode Mode of the tensor to contract with the matrix
 * @param transpose Whether to transpose the matrix before multiplication
 * @return Resulting tensor after multiplication
 */
// [[Rcpp::export]]
NumericVector ttm_cpp(const NumericVector &tensor_data,
                      const NumericMatrix &matrix, int mode,
                      bool transpose = false) {
  try {
    const std::size_t axis = static_cast<std::size_t>(mode - 1);
    std::vector<std::size_t> tensor_shape = tensory::array_dims(tensor_data);
    const std::size_t n_dim = tensor_shape.size();

    if (axis >= n_dim) {
      Rcpp::stop("mode is out of bounds for the tensor order");
    }

    const std::size_t Ik = tensor_shape[axis];
    const std::size_t J = transpose ? static_cast<std::size_t>(matrix.ncol())
                                    : static_cast<std::size_t>(matrix.nrow());

    std::vector<std::size_t> final_shape = tensor_shape;
    final_shape[axis] = J;

    const std::size_t total_size = product_of_dims(tensor_shape);
    const std::size_t rest = total_size / Ik;

    auto input_strides = column_major_strides(tensor_shape);
    auto output_strides = column_major_strides(final_shape);

    std::vector<std::size_t> other_modes;
    std::vector<std::size_t> other_dims;
    other_modes.reserve(n_dim > 0 ? n_dim - 1 : 0);
    other_dims.reserve(n_dim > 0 ? n_dim - 1 : 0);
    for (std::size_t i = 0; i < n_dim; ++i) {
      if (i == axis) {
        continue;
      }
      other_modes.push_back(i);
      other_dims.push_back(tensor_shape[i]);
    }

    std::vector<double> x_mat(total_size);
    const double *tensor_ptr = REAL(tensor_data);
    const std::size_t axis_input_stride = input_strides[axis];

    for (std::size_t rest_index = 0; rest_index < rest; ++rest_index) {
      std::size_t tmp = rest_index;
      std::size_t base_input = 0;

      for (std::size_t p = 0; p < other_modes.size(); ++p) {
        const std::size_t coord = tmp % other_dims[p];
        tmp /= other_dims[p];
        base_input += coord * input_strides[other_modes[p]];
      }

      const std::size_t column_offset = Ik * rest_index;
      for (std::size_t k_idx = 0; k_idx < Ik; ++k_idx) {
        x_mat[column_offset + k_idx] = tensor_ptr[base_input + k_idx * axis_input_stride];
      }
    }

    std::vector<double> y_mat(J * rest);

    const std::size_t int_max = static_cast<std::size_t>(INT_MAX);
    if (J > int_max || rest > int_max || Ik > int_max) {
      Rcpp::stop("Tensor extents exceed BLAS 32-bit integer limit");
    }

    const char *transa = transpose ? "T" : "N";
    const char *transb = "N";
    const int m = static_cast<int>(J);
    const int n = static_cast<int>(rest);
    const int k = static_cast<int>(Ik);
    const double alpha = 1.0;
    const double beta = 0.0;
    const int lda = static_cast<int>(transpose ? Ik : J);
    const int ldb = static_cast<int>(Ik);
    const int ldc = static_cast<int>(J);
    const double *mat_ptr = REAL(matrix);

    F77_NAME(dgemm)(transa, transb, &m, &n, &k, &alpha, mat_ptr, &lda,
                    x_mat.data(), &ldb, &beta, y_mat.data(), &ldc FCONE FCONE);

    NumericVector result_tensor = tensory::alloc_array(final_shape);
    double *result_ptr = REAL(result_tensor);
    const std::size_t axis_output_stride = output_strides[axis];

    for (std::size_t rest_index = 0; rest_index < rest; ++rest_index) {
      std::size_t tmp = rest_index;
      std::size_t base_output = 0;

      for (std::size_t p = 0; p < other_modes.size(); ++p) {
        const std::size_t coord = tmp % other_dims[p];
        tmp /= other_dims[p];
        base_output += coord * output_strides[other_modes[p]];
      }

      const std::size_t column_offset = J * rest_index;
      for (std::size_t j_idx = 0; j_idx < J; ++j_idx) {
        result_ptr[base_output + j_idx * axis_output_stride] = y_mat[column_offset + j_idx];
      }
    }

    return result_tensor;
  } catch (const std::exception &e) {
    Rcpp::stop("Error in optimized ttm_cpp: " + std::string(e.what()));
  }
}

// [[Rcpp::export]]
NumericVector ttm_multiple_cpp(const NumericVector &tensor_data,
                               const List &matrices,
                               const IntegerVector &modes,
                               bool transpose = false) {
  try {
    if (matrices.size() == 0) {
      Rcpp::stop("Empty list of matrices provided");
    }

    NumericMatrix first_matrix = as<NumericMatrix>(matrices[0]);
    NumericVector result = ttm_cpp(tensor_data, first_matrix, modes[0], transpose);

    for (R_xlen_t i = 1; i < matrices.size(); ++i) {
      NumericMatrix matrix = as<NumericMatrix>(matrices[i]);
      result = ttm_cpp(result, matrix, modes[i], transpose);
    }

    return result;

  } catch (const std::exception &e) {
    Rcpp::stop("Error in ttm_multiple_cpp: " + std::string(e.what()));
  }
}
