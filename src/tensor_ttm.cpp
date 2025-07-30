#include "xtensor-r/rarray.hpp"
#include "xtensor/containers/xtensor.hpp"
#include "xtensor/containers/xarray.hpp"
#include "xtensor/misc/xmanipulation.hpp"
#include "xtensor-blas/xlinalg.hpp"
#include <Rcpp.h>
#include <vector>
#include <numeric>
#include <algorithm>

using namespace Rcpp;

/**
 * @brief Helper function to convert R matrix to xtensor efficiently
 * @param matrix R NumericMatrix
 * @return xtensor rarray
 */
inline xt::rarray<double> convert_r_matrix_to_xtensor(const NumericMatrix& matrix) {
    std::vector<size_t> dims = {static_cast<size_t>(matrix.nrow()), 
                               static_cast<size_t>(matrix.ncol())};
    xt::rarray<double> result = xt::zeros<double>(dims);
    std::copy(matrix.begin(), matrix.end(), result.begin());
    return result;
}

/**
 * @brief Helper function to create permutation vector for dimension reordering
 * @param N Total number of dimensions
 * @param axis Position where new dimension should be placed
 * @return Permutation vector
 */
inline std::vector<std::size_t> create_permutation(std::size_t N, std::size_t axis) {
    std::vector<std::size_t> permutation(N);
    
    // Fill indices before axis
    std::iota(permutation.begin(), permutation.begin() + axis, 0);
    
    // Place new dimension (originally at position N-1) at axis position
    permutation[axis] = N - 1;
    
    // Fill remaining indices, shifted down by 1
    std::iota(permutation.begin() + axis + 1, permutation.end(), axis);
    
    return permutation;
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
                          const xt::rarray<double>& matrix_data,
                          int mode,
                          bool transpose = false) {
    try {
        const std::size_t axis = static_cast<std::size_t>(mode - 1);
        
        // Avoid unnecessary copies by using conditional logic
        if (transpose) {
            // Use transpose view - no copy needed
            const auto temp_result = xt::linalg::tensordot(tensor_data, matrix_data, {axis}, {0});
            
            // Create permutation and apply
            const auto permutation = create_permutation(temp_result.dimension(), axis);
            const auto transposed_result = xt::transpose(temp_result, permutation);
            
            // Apply squeeze to remove singleton dimensions
            return xt::eval(transposed_result);
            
        } else {
            // Direct computation - no copy needed
            const auto temp_result = xt::linalg::tensordot(tensor_data, matrix_data, {axis}, {1});
            
            // Create permutation and apply
            const auto permutation = create_permutation(temp_result.dimension(), axis);
            const auto transposed_result = xt::transpose(temp_result, permutation);
            
            // Apply squeeze to remove singleton dimensions
            return xt::eval(transposed_result);
        }
        
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
        // Pre-convert all matrices to avoid repeated conversions
        std::vector<xt::rarray<double>> converted_matrices;
        converted_matrices.reserve(matrices.size());
        
        for (const auto& matrix : matrices) {
            converted_matrices.emplace_back(convert_r_matrix_to_xtensor(as<NumericMatrix>(matrix)));
        }
        
        // Start with input tensor
        auto result = tensor_data;
        
        // Apply operations sequentially
        for (std::size_t i = 0; i < converted_matrices.size(); ++i) {
            result = ttm_cpp(result, converted_matrices[i], modes[i], transpose);
        }
        
        return result;
        
    } catch (const std::exception& e) {
        Rcpp::stop("Error in ttm_multiple_cpp: " + std::string(e.what()));
    }
}
