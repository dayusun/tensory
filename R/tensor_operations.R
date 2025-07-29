#' Tensor Times Matrix/Vector (ttm) Operation
#'
#' Compute a tensor times a matrix (or matrices) or vector (or vectors) in one (or more) modes.
#' This function implements the tensor times matrix/vector operation similar to MATLAB's ttm/ttv functions.
#' Uses efficient Xtensor-blas implementation for high-performance tensor operations.
#'
#' The tensor times matrix (ttm) operation multiplies a tensor by a matrix along a specified mode.
#' For a tensor X of size I1 x I2 x ... x IN and matrix M of size J x Ik, the result is a tensor
#' of size I1 x ... x I(k-1) x J x I(k+1) x ... x IN.
#'
#' The tensor times vector (ttv) operation multiplies a tensor by a vector along a specified mode,
#' reducing the dimensionality by one. For a tensor X of size I1 x I2 x ... x IN and vector v of 
#' length Ik, the result is a tensor of size I1 x ... x I(k-1) x I(k+1) x ... x IN.
#'
#' @param tensor A Tensor object
#' @param matrix A matrix, vector, or list of matrices/vectors to multiply with the tensor
#' @param mode Integer specifying which mode (dimension) to multiply. 
#'             For single matrix/vector, defaults to 1. For multiple matrices/vectors, 
#'             this should be a vector of modes corresponding to each matrix/vector.
#' @param transpose Logical. If TRUE, transpose the matrix before multiplication.
#'                  Default is FALSE. (Ignored for vectors)
#' @return A new Tensor object with the result of the multiplication
#'
#' @examples
#' # Create a 3D tensor
#' t <- tensor(array(1:24, dim = c(4, 3, 2)))
#' 
#' # Create a matrix
#' m <- matrix(1:8, nrow = 4, ncol = 2)
#' 
#' # Multiply tensor by matrix in mode 1
#' result <- ttm(t, m, mode = 1)
#' 
#' # Multiple matrices
#' m1 <- matrix(1:8, nrow = 4, ncol = 2)
#' m2 <- matrix(1:6, nrow = 3, ncol = 2)
#' result <- ttm(t, list(m1, m2), mode = c(1, 2))
#' 
#' # Vector multiplication (tensor times vector)
#' v <- 1:4
#' result <- ttm(t, v, mode = 1)
#' 
#' # Multiple vectors
#' v1 <- 1:4
#' v2 <- 1:3
#' result <- ttm(t, list(v1, v2), mode = c(1, 2))
#'
#' @export
ttm <- function(tensor, matrix, mode = 1, transpose = FALSE) {
  # Manual dispatch for R6 Tensor class
  if (!inherits(tensor, "Tensor")) {
    stop("ttm is not implemented for this object type")
  }
  
  if (is.matrix(matrix) || (is.array(matrix) && length(dim(matrix)) == 2)) {
    # Single matrix case
    matrix_data <- matrix
    if (!is.matrix(matrix_data)) {
      matrix_data <- as.matrix(matrix_data)
    }
    
    # Validate mode
    tensor_dims <- tensor$dim()
    if (mode < 1 || mode > length(tensor_dims)) {
      stop("Mode must be between 1 and number of tensor dimensions")
    }
    
    # Validate matrix dimensions - the contracted dimension must match tensor dimension
    tensor_mode_dim <- tensor_dims[mode]
    # When transpose=TRUE, matrix gets transposed internally in C++
    # So we validate against what will actually be contracted
    # transpose=FALSE: contract with matrix columns (new behavior)
    # transpose=TRUE: contract with matrix rows of transposed matrix = matrix columns of original
    contracted_dim <- if (transpose) ncol(matrix_data) else ncol(matrix_data)
    
    if (contracted_dim != tensor_mode_dim) {
      stop(paste0("Matrix columns (", contracted_dim, ") must match tensor dimension size (", 
                  tensor_mode_dim, ") for mode ", mode))
    }
    
    # Call C++ function for single matrix multiplication
    result_data <- ttm_cpp(tensor$as_array(), matrix_data, mode, transpose)
    return(Tensor$new(result_data))
    
  } else if (is.vector(matrix) || is.numeric(matrix)) {
    # Single vector case (tensor times vector)
    vector_data <- as.vector(matrix)
    
    # Validate mode
    tensor_dims <- tensor$dim()
    if (mode < 1 || mode > length(tensor_dims)) {
      stop("Mode must be between 1 and number of tensor dimensions")
    }
    
    # Validate vector length
    tensor_mode_dim <- tensor_dims[mode]
    if (length(vector_data) != tensor_mode_dim) {
      stop(paste0("Vector length (", length(vector_data), ") must match tensor dimension size (", 
                  tensor_mode_dim, ") for mode ", mode))
    }
    
    # For tensor times vector, we need to convert vector to 1 x n matrix
    # and use transpose=TRUE so that we contract with columns (the vector elements)
    vector_matrix <- matrix(vector_data, nrow = 1, ncol = length(vector_data))
    
    # Call ttm with transpose=TRUE to contract with matrix columns (vector elements)
    result_tensor <- ttm(tensor, vector_matrix, mode = mode, transpose = TRUE)
    
    # Squeeze the dimension that became size 1
    result_dims <- result_tensor$dim()
    new_dims <- result_dims[result_dims != 1]
    
    if (length(new_dims) == 0) {
      # Scalar result
      return(Tensor$new(as.vector(result_tensor$as_array()), 1))
    } else {
      # Reshape to remove singleton dimensions
      return(result_tensor$reshape(new_dims))
    }
    
  } else if (is.list(matrix)) {
    # Multiple matrices/vectors case - integrate ttm_multiple functionality directly
    matrices <- matrix  # Rename for clarity
    
    # Validate input
    if (!is.list(matrices)) {
      stop("Matrices/vectors must be provided as a list")
    }
    
    num_matrices <- length(matrices)
    
    # If modes not provided, use sequential modes starting from 1
    if (missing(mode) || is.null(mode)) {
      modes <- 1:num_matrices
    } else {
      modes <- mode
    }
    
    # Validate modes
    if (length(modes) != num_matrices) {
      stop("Number of modes must match number of matrices/vectors")
    }
    
    # Check if all elements are matrices or all are vectors
    all_matrices <- all(sapply(matrices, function(x) is.matrix(x) || (is.array(x) && length(dim(x)) == 2)))
    all_vectors <- all(sapply(matrices, function(x) is.vector(x) && !is.matrix(x) && !is.array(x)))
    
    if (all_matrices) {
      # All matrices case
      # Validate that all elements in matrices list are actually matrices
      for (i in seq_along(matrices)) {
        if (!is.matrix(matrices[[i]]) && !(is.array(matrices[[i]]) && length(dim(matrices[[i]])) == 2)) {
          stop(paste0("Element ", i, " in matrices list is not a matrix"))
        }
      }
      
      # Call C++ function for multiple matrix multiplication
      result_data <- ttm_multiple_cpp(tensor$as_array(), matrices, modes, transpose)
      return(Tensor$new(result_data))
      
    } else if (all_vectors) {
      # All vectors case (multiple tensor times vector)
      # Apply each vector multiplication sequentially
      result_tensor <- tensor$clone_tensor()
      
      # Apply vectors in the order they were provided
      # Need to adjust modes as dimensions reduce
      adjusted_modes <- modes
      for (i in seq_along(matrices)) {
        # Apply the current vector multiplication
        current_mode <- adjusted_modes[i]
        result_tensor <- ttm(result_tensor, matrices[[i]], mode = current_mode)
        
        # Adjust remaining modes for dimension reduction
        if (i < length(matrices)) {
          for (j in (i+1):length(matrices)) {
            if (adjusted_modes[j] > current_mode) {
              adjusted_modes[j] <- adjusted_modes[j] - 1
            }
          }
        }
      }
      
      return(result_tensor)
      
    } else {
      stop("List must contain either all matrices or all vectors")
    }
    
  } else {
    stop("Input must be a matrix, vector, or list of matrices/vectors")
  }
}
