#include "xtensor-r/rarray.hpp"
#include "xtensor/containers/xtensor.hpp"
#include "xtensor/containers/xarray.hpp"
#include "xtensor/misc/xmanipulation.hpp"
#include "xtensor-blas/xlinalg.hpp"
#include <Rcpp.h>
#include <vector>

using namespace Rcpp;

/**
 * @brief Performs a mode-k product of a tensor and a matrix.
 * Contracts tensor dimension (k-1) with matrix columns (dimension 1).
 * Result replaces tensor dimension (k-1) with matrix rows (dimension 0).
 *
 * @param tensor The input xtensor.
 * @param matrix The matrix to multiply with.
 * @param k The mode (1-based) of the tensor to multiply along.
 * @return The resulting tensor.
 */
template <class T, class M>
auto ttm_mode_k(const T& tensor, const M& matrix, std::size_t k) {
    // k is 1-based, axis is 0-based
    std::size_t axis = k - 1;

    // Contract tensor axis (k-1) with matrix columns (axis 1)
    auto temp_result = xt::linalg::tensordot(tensor, matrix, {axis}, {1});

    // Create the permutation vector to move the new dimension to the correct spot
    auto N = temp_result.dimension();
    std::vector<std::size_t> permutation(N);
    
    // New dimension is at position N-1 (last position)
    // We want to move it to position axis
    for (std::size_t i = 0; i < axis; ++i) {
        permutation[i] = i;
    }
    permutation[axis] = N - 1;
    for (std::size_t i = axis + 1; i < N; ++i) {
        permutation[i] = i - 1;
    }

    return xt::eval(xt::transpose(temp_result, permutation));
}

// [[Rcpp::export]]
xt::rarray<double> ttm_cpp(const xt::rarray<double>& tensor_data, 
                          const xt::rarray<double>& matrix_data,
                          int mode,
                          bool transpose = false) {
    try {
        // Get tensor dimensions
        auto tensor_dims = tensor_data.shape();
        auto matrix_dims = matrix_data.shape();
        
        // Validate mode
        if (mode < 1 || mode > static_cast<int>(tensor_dims.size())) {
            Rcpp::stop("Mode must be between 1 and number of tensor dimensions");
        }
        
        int mode_idx = mode - 1; // Convert to 0-based indexing
        
        // Get the dimension size for the specified mode
        size_t tensor_mode_dim = tensor_dims[mode_idx];
        
        // Handle transpose
        xt::rarray<double> effective_matrix = matrix_data;
        int matrix_contract_axis = 1; // Default: contract with matrix columns
        
        if (transpose) {
            // Transpose the matrix
            effective_matrix = xt::transpose(matrix_data);
            matrix_contract_axis = 0; // After transpose, contract with matrix rows (original columns)
        }
        
        // Validate matrix dimensions - contracted dimension must match tensor dimension
        size_t contracted_dim = effective_matrix.shape()[matrix_contract_axis];
        if (contracted_dim != tensor_mode_dim) {
            std::string axis_name = (matrix_contract_axis == 0) ? "rows" : "columns";
            std::string error_msg = "Matrix " + axis_name + " (" + std::to_string(contracted_dim) + 
                                  ") must match tensor dimension size (" + std::to_string(tensor_mode_dim) + 
                                  ") for the specified mode";
            Rcpp::stop(error_msg);
        }
        
        // Perform mode-k product with correct contraction axis
        // k is 1-based, axis is 0-based
        std::size_t axis = mode - 1;
        std::size_t contract_axis = static_cast<std::size_t>(matrix_contract_axis);
        auto temp_result = xt::linalg::tensordot(tensor_data, effective_matrix, {axis}, {contract_axis});

        // Create the permutation vector to move the new dimension to the correct spot
        auto N = temp_result.dimension();
        std::vector<std::size_t> permutation(N);
        
        // New dimension is at position N-1 (last position)
        // We want to move it to position axis
        for (std::size_t i = 0; i < axis; ++i) {
            permutation[i] = i;
        }
        permutation[axis] = N - 1;
        for (std::size_t i = axis + 1; i < N; ++i) {
            permutation[i] = i - 1;
        }

        auto result = xt::eval(xt::transpose(temp_result, permutation));
        
        return result;
        
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
        // Start with the input tensor
        auto result = xt::eval(tensor_data);
        
        // Apply each matrix multiplication sequentially
        for (int i = 0; i < matrices.size(); i++) {
            NumericMatrix matrix = as<NumericMatrix>(matrices[i]);
            
            // Convert R matrix to xtensor
            std::vector<size_t> matrix_dims = {static_cast<size_t>(matrix.nrow()), 
                                              static_cast<size_t>(matrix.ncol())};
            xt::rarray<double> xmatrix = xt::zeros<double>(matrix_dims);
            std::copy(matrix.begin(), matrix.end(), xmatrix.begin());
            
            // Handle transpose
            xt::rarray<double> effective_matrix = xmatrix;
            int matrix_contract_axis = 1; // Default: contract with matrix columns
            
            if (transpose) {
                effective_matrix = xt::transpose(xmatrix);
                matrix_contract_axis = 0; // After transpose, contract with matrix rows (original columns)
            }
            
            // Get current result dimensions
            auto result_dims = result.shape();
            int mode = modes[i];
            
            // Validate mode
            if (mode < 1 || mode > static_cast<int>(result_dims.size())) {
                Rcpp::stop("Mode must be between 1 and number of tensor dimensions");
            }
            
            int mode_idx = mode - 1; // Convert to 0-based indexing
            size_t tensor_mode_dim = result_dims[mode_idx];
            
            // Validate matrix dimensions - contracted dimension must match tensor dimension
            size_t contracted_dim = effective_matrix.shape()[matrix_contract_axis];
            if (contracted_dim != tensor_mode_dim) {
                std::string axis_name = (matrix_contract_axis == 0) ? "rows" : "columns";
                std::string error_msg = "Matrix " + axis_name + " (" + std::to_string(contracted_dim) + 
                                      ") must match tensor dimension size (" + std::to_string(tensor_mode_dim) + 
                                      ") for the specified mode";
                Rcpp::stop(error_msg);
            }
            
            // Perform mode-k product with correct contraction axis
            // k is 1-based, axis is 0-based
            std::size_t axis = mode - 1;
            std::size_t contract_axis = static_cast<std::size_t>(matrix_contract_axis);
            auto temp_result = xt::linalg::tensordot(result, effective_matrix, {axis}, {contract_axis});

            // Create the permutation vector to move the new dimension to the correct spot
            auto N = temp_result.dimension();
            std::vector<std::size_t> permutation(N);
            
            // New dimension is at position N-1 (last position)
            // We want to move it to position axis
            for (std::size_t i = 0; i < axis; ++i) {
                permutation[i] = i;
            }
            permutation[axis] = N - 1;
            for (std::size_t i = axis + 1; i < N; ++i) {
                permutation[i] = i - 1;
            }

            result = xt::eval(xt::transpose(temp_result, permutation));
        }
        
        return result;
        
    } catch (const std::exception& e) {
        Rcpp::stop("Error in ttm_multiple_cpp: " + std::string(e.what()));
    }
}
