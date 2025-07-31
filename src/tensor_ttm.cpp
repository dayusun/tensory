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
 * @brief Helper function to create zero-copy R-aware matrix view using xt::rarray
 * @param matrix R NumericMatrix  
 * @return xtensor rarray view that directly maps R memory with correct column-major layout
 */
inline auto create_matrix_view(const NumericMatrix& matrix) {
    // Create zero-copy rarray directly from R SEXP - this preserves R's column-major layout
    return xt::rarray<double>(SEXP(matrix));
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
                          const NumericMatrix& matrix,
                          int mode,
                          bool transpose = false) {
    try {
        const std::size_t axis = static_cast<std::size_t>(mode - 1);
        
        // Create zero-copy view of the matrix
        auto matrix_view = create_matrix_view(matrix);
        
        // Use optimized zero-copy matrix view  
        if (transpose) {
            // Use transpose view - no copy needed
            const auto temp_result = xt::linalg::tensordot(tensor_data, matrix_view, {axis}, {0});
            
            // Create permutation and apply
            const auto permutation = create_permutation(temp_result.dimension(), axis);
            const auto transposed_result = xt::transpose(temp_result, permutation);
            
            // Apply squeeze to remove singleton dimensions
            return xt::eval(transposed_result);
            
        } else {
            // Direct computation - no copy needed
            const auto temp_result = xt::linalg::tensordot(tensor_data, matrix_view, {axis}, {1});
            
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
