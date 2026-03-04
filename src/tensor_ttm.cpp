#include "xtensor-r/rarray.hpp"
#include "xtensor/containers/xtensor.hpp"
#include "xtensor/containers/xarray.hpp"
#include "xtensor/containers/xadapt.hpp"
#include "xtensor/misc/xmanipulation.hpp"
#include "xtensor-blas/xlinalg.hpp"
#include <Rcpp.h>
#include <vector>
#include <numeric>
#include <algorithm>
#include <array>
#include <cstring>

// Use R's BLAS interface via Fortran calls (standard R package approach)
// Fortran name mangling
#ifndef F77_NAME
#define F77_NAME(x) x ## _
#endif

#ifndef FCONE
#define FCONE
#endif

// Externally declare the Fortran BLAS subroutines
extern "C" {
    void F77_NAME(dgemm)(const char* transa, const char* transb, const int* m, const int* n, const int* k,
                         const double* alpha, const double* a, const int* lda, const double* b, const int* ldb,
                         const double* beta, double* c, const int* ldc FCONE FCONE);
}

using namespace Rcpp;

/**
 * @brief Helper function to create zero-copy R-aware matrix view using xt::rarray
 * @param matrix R NumericMatrix  
 * @return xtensor rarray view that directly maps R memory with correct column-major layout
 */
inline auto create_matrix_view(const NumericMatrix& matrix) {
  // Create zero-copy rarray directly from R SEXP - this preserves R's column-major layout
  return xt::rarray<double>(SEXP(matrix));
}






/**
 * @brief Tensor-times-matrix operation with optional transpose
 * @param tensor_data Input tensor as xtensor rarray
 * @param matrix_data Input matrix as xtensor rarray
 * @param mode Mode of the tensor to contract with the matrix
 * @param transpose Whether to transpose the matrix before multiplication
 * @return Resulting tensor after multiplication
 */
// [[Rcpp::export]]
xt::rarray<double> ttm_cpp(const xt::rarray<double>& tensor_data,
                          const NumericMatrix& matrix,
                          int mode,
                          bool transpose = false) {
    try {
        const std::size_t axis = static_cast<std::size_t>(mode - 1);
        auto tensor_shape = tensor_data.shape();
        std::size_t n_dim = tensor_data.dimension();

        // Matrix dimensions
        const int M_nrow = matrix.nrow();
        const int M_ncol = matrix.ncol();

        // Calculate contraction parameters
        std::size_t contract_len, new_dim;
        if (transpose) {
            contract_len = static_cast<std::size_t>(M_nrow);
            new_dim = static_cast<std::size_t>(M_ncol);
        } else {
            contract_len = static_cast<std::size_t>(M_ncol);
            new_dim = static_cast<std::size_t>(M_nrow);
        }

        // OPTIMIZATION 1: Create transpose permutation more efficiently
        std::vector<std::size_t> tensor_transpose_axes;
        tensor_transpose_axes.reserve(n_dim);
        
        for (std::size_t i = 0; i < n_dim; ++i) {
            if (i < axis) {
                tensor_transpose_axes.push_back(i + 1);
            } else if (i == axis) {
                tensor_transpose_axes.push_back(0);
            } else {
                tensor_transpose_axes.push_back(i);
            }
        }
        
        // OPTIMIZATION 2: Use auto to avoid forced layout conversion
        auto tensor_transposed = xt::transpose(tensor_data, tensor_transpose_axes);

        // Calculate keep dimensions
        std::size_t keep_len = 1;
        for (std::size_t i = 1; i < n_dim; ++i) {
            keep_len *= tensor_transposed.shape()[i];
        }

        // OPTIMIZATION 3: Create contiguous tensor matrix (avoid reshape on view)
        xt::xarray<double, xt::layout_type::column_major> tensor_matrix = tensor_transposed;
        tensor_matrix.reshape({contract_len, keep_len});

        // OPTIMIZATION 4: Pre-allocate result with correct constructor
        std::vector<std::size_t> result_shape = {new_dim, keep_len};
        xt::xarray<double, xt::layout_type::column_major> result_matrix(result_shape);

        // OPTIMIZATION 5: Optimized BLAS parameters
        const char* transa = transpose ? "T" : "N";
        const char* transb = "N";
        const int m = static_cast<int>(new_dim);
        const int n = static_cast<int>(keep_len);
        const int k = static_cast<int>(contract_len);
        const double alpha = 1.0;
        const double beta = 0.0;

        // Use optimized leading dimensions
        const int lda = M_nrow;
        const int ldb = static_cast<int>(contract_len);
        const int ldc = static_cast<int>(new_dim);

        // BLAS multiplication
        F77_NAME(dgemm)(transa, transb, &m, &n, &k, &alpha,
                       REAL(matrix), &lda, tensor_matrix.data(), &ldb,
                       &beta, result_matrix.data(), &ldc FCONE FCONE);

        // OPTIMIZATION 6: Efficient result reshaping
        std::vector<std::size_t> final_shape;
        final_shape.reserve(n_dim);
        final_shape.push_back(new_dim);
        for (std::size_t i = 0; i < n_dim; ++i) {
            if (i != axis) {
                final_shape.push_back(tensor_shape[i]);
            }
        }
        result_matrix.reshape(final_shape);

        // OPTIMIZATION 7: Efficient inverse transpose with direct indexing
        std::vector<std::size_t> inverse_axes;
        inverse_axes.reserve(n_dim);
        inverse_axes.push_back(axis);
        
        for (std::size_t i = 1; i < n_dim; ++i) {
            if (i <= axis) {
                inverse_axes.push_back(i - 1);
            } else {
                inverse_axes.push_back(i);
            }
        }

        // Apply inverse transpose to restore original dimension order
        auto result_final = xt::transpose(result_matrix, inverse_axes);
        
        return xt::rarray<double>(result_final);
        
    } catch (const std::exception& e) {
        Rcpp::stop("Error in ttm_cpp: " + std::string(e.what()));
    }
}

// [[Rcpp::export]]
xt::rarray<double> ttm_multiple_cpp(const xt::rarray<double>& tensor_data,
                                   const List& matrices,
                                   const IntegerVector& modes,
                                   bool transpose = false) {
    try {
        if (matrices.size() == 0) {
            Rcpp::stop("Empty list of matrices provided");
        }
        
        // Create zero-copy matrix views instead of converting to xtensor
        std::vector<decltype(create_matrix_view(std::declval<NumericMatrix>()))> matrix_views;
        matrix_views.reserve(matrices.size());
        
        for (const auto& matrix : matrices) {
            matrix_views.emplace_back(create_matrix_view(as<NumericMatrix>(matrix)));
        }
        
        // Create list of modes and matrices, sorted by mode in descending order
        std::vector<std::pair<int, size_t>> mode_index_pairs;
        for (size_t i = 0; i < matrix_views.size(); ++i) {
            mode_index_pairs.emplace_back(modes[i], i);
        }
        
        // Sort by mode in descending order for efficient contraction
        std::sort(mode_index_pairs.begin(), mode_index_pairs.end(), 
                  [](const auto& a, const auto& b) { return a.first > b.first; });
        
        // Perform all contractions sequentially without intermediate transposes
        auto result = tensor_data;
        const size_t original_rank = tensor_data.dimension();
        
        for (const auto& pair : mode_index_pairs) {
            const int mode = pair.first;
            const size_t matrix_idx = pair.second;
            const size_t axis = static_cast<size_t>(mode - 1);
            
            if (transpose) {
                result = xt::linalg::tensordot(result, matrix_views[matrix_idx], {axis}, {0});
            } else {
                result = xt::linalg::tensordot(result, matrix_views[matrix_idx], {axis}, {1});
            }
        }
        
        // Calculate final permutation to restore natural axis order
        std::vector<int> original_modes(original_rank);
        std::iota(original_modes.begin(), original_modes.end(), 0);
        
        // Get multiplied modes (0-based) in sorted order
        std::vector<int> multiplied_modes;
        for (int i = 0; i < modes.size(); ++i) {
            multiplied_modes.push_back(modes[i] - 1);  // Convert to 0-based
        }
        std::sort(multiplied_modes.begin(), multiplied_modes.end());
        
        // Find remaining modes (not multiplied)
        std::vector<int> remaining_modes;
        std::set_difference(original_modes.begin(), original_modes.end(),
                          multiplied_modes.begin(), multiplied_modes.end(),
                          std::back_inserter(remaining_modes));
        
        // Get multiplied modes in the order they appear in result (descending)
        std::vector<int> multiplied_modes_desc;
        for (const auto& pair : mode_index_pairs) {
            multiplied_modes_desc.push_back(pair.first - 1);  // Convert to 0-based
        }
        
        // Current layout: (remaining_modes..., multiplied_modes_desc...)
        std::vector<int> current_layout;
        current_layout.insert(current_layout.end(), remaining_modes.begin(), remaining_modes.end());
        current_layout.insert(current_layout.end(), multiplied_modes_desc.begin(), multiplied_modes_desc.end());
        
        // Calculate permutation to restore natural order (0, 1, 2, ...)
        std::vector<size_t> final_permutation(original_rank);
        for (size_t i = 0; i < original_rank; ++i) {
            auto it = std::find(current_layout.begin(), current_layout.end(), static_cast<int>(i));
            final_permutation[i] = std::distance(current_layout.begin(), it);
        }
        
        // Apply single final transpose to restore axis order
        return xt::eval(xt::transpose(result, final_permutation));
        
    } catch (const std::exception& e) {
        Rcpp::stop("Error in ttm_multiple_cpp: " + std::string(e.what()));
    }
}
