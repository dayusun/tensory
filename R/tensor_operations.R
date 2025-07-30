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

    # Validate mode (ensure it's a single value)
    if (length(mode) != 1) {
      stop("Mode must be a single integer for single matrix operations")
    }

    tensor_dims <- tensor$dim()
    if (mode < 1 || mode > length(tensor_dims)) {
      stop("Mode must be between 1 and number of tensor dimensions")
    }

    # Validate matrix dimensions - the contracted dimension must match tensor dimension
    tensor_mode_dim <- tensor_dims[mode]

    # Validate dimensions based on transpose flag
    if (transpose) {
      # When transpose=TRUE: matrix rows become the contracted dimension
      contracted_dim <- nrow(matrix)
      if (contracted_dim != tensor_mode_dim) {
        stop(paste0(
          "Matrix rows (", contracted_dim,
          ") must match tensor dimension size (", tensor_mode_dim,
          ") for mode ", mode, " with transpose=TRUE"
        ))
      }
    } else {
      # When transpose=FALSE: contract with matrix columns
      contracted_dim <- ncol(matrix)
      if (contracted_dim != tensor_mode_dim) {
        stop(paste0(
          "Matrix columns (", contracted_dim,
          ") must match tensor dimension size (", tensor_mode_dim,
          ") for mode ", mode
        ))
      }
    }

    # Ensure matrix is double precision before passing to C++
    if (storage.mode(matrix) != "double") {
      storage.mode(matrix) <- "double"
    }

    # Call C++ function for single matrix multiplication
    result_data <- ttm_cpp(tensor$data, matrix, mode, transpose)
    return(Tensor$new(result_data))
  } else if ((is.vector(matrix) || is.numeric(matrix)) && !is.list(matrix)) {
    # Single vector case (tensor times vector)

    # Validate mode (ensure it's a single value)
    if (length(mode) != 1) {
      stop("Mode must be a single integer for single vector operations")
    }

    tensor_dims <- tensor$dim()
    if (mode < 1 || mode > length(tensor_dims)) {
      stop("Mode must be between 1 and number of tensor dimensions")
    }

    # Validate vector length
    tensor_mode_dim <- tensor_dims[mode]
    if (length(matrix) != tensor_mode_dim) {
      stop(paste0(
        "Vector length (", length(matrix), ") must match tensor dimension size (",
        tensor_mode_dim, ") for mode ", mode
      ))
    }

    # Ensure vector is double precision
    if (storage.mode(matrix) != "double") {
      storage.mode(matrix) <- "double"
    }

    # For tensor times vector, we need to convert vector to matrix
    # Create row vector (1 x n) and use transpose=FALSE to contract with columns
    vector_matrix <- matrix(matrix, nrow = 1)

    # Call ttm with transpose=FALSE to contract with matrix columns (vector elements)
    result_tensor <- ttm(tensor, vector_matrix, mode = mode, transpose = FALSE)

    # Remove the singleton dimension that was introduced
    result_dims <- result_tensor$dim()
    if (length(result_dims) == 1 && result_dims[1] == 1) {
      # Scalar result
      return(Tensor$new(as.vector(result_tensor$data), c()))
    } else {
      # Remove singleton dimensions (dimensions of size 1)
      new_dims <- result_dims[result_dims != 1]
      if (length(new_dims) == 0) {
        # All dimensions were 1, result is scalar
        return(Tensor$new(as.vector(result_tensor$data), c()))
      } else {
        # Reshape to remove singleton dimensions
        return(result_tensor$reshape(new_dims))
      }
    }
  } else if (is.list(matrix)) {
    # Multiple matrices/vectors case - integrate ttm_multiple functionality directly

    # Validate input
    if (length(matrix) == 0) {
      stop("Empty list of matrices/vectors provided")
    }

    num_matrices <- length(matrix)

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

    # Validate that modes are within valid range
    tensor_dims <- tensor$dim()
    if (any(modes < 1) || any(modes > length(tensor_dims))) {
      stop("All modes must be between 1 and number of tensor dimensions")
    }

    # Check if all elements are matrices (vectors are not allowed in this case)
    element_types <- sapply(matrix, function(x) {
      if (is.matrix(x) || (is.array(x) && length(dim(x)) == 2)) {
        "matrix"
      } else if (is.vector(x) && !is.matrix(x) && !is.array(x)) {
        "vector"
      } else {
        "invalid"
      }
    })

    # Check for invalid types
    if (any(element_types == "invalid")) {
      invalid_indices <- which(element_types == "invalid")
      stop(paste0(
        "Invalid elements at positions: ", paste(invalid_indices, collapse = ", "),
        ". All elements must be matrices."
      ))
    }

    # Convert vectors to matrices if any vectors are present
    for (i in seq_along(matrix)) {
      if (element_types[i] == "vector") {
        matrix[[i]] <- matrix(matrix[[i]], nrow = 1)  # Convert to row vector
      }
    }

    # Ensure all matrices are double precision before passing to C++
    for (i in seq_along(matrix)) {
      if (storage.mode(matrix[[i]]) != "double") {
        storage.mode(matrix[[i]]) <- "double"
      }
    }

    # All matrices case - validate dimensions before calling C++

    if (transpose) {
      # If transpose is TRUE, we need to validate the number of rows in each matrix
      # as they will be contracted with tensor dimensions
      matrice_mode <- sapply(matrix, nrow)
    } else {
      matrice_mode <- sapply(matrix, ncol)
    }

    # Validate matrix dimensions for current tensor state
    tensor_mode_dim <- tensor_dims[modes]

    if (any(tensor_mode_dim != matrice_mode)) {
      unmatched_modes_position <- which(tensor_mode_dim != matrice_mode)
      axis_name <- if (transpose) "rows" else "columns"
      stop(paste0(
        "Matrix ", paste0(unmatched_modes_position, ", "), " ", axis_name, " (",
        matrice_mode[unmatched_modes_position],
        ") must match tensor dimension size (", tensor_mode_dim[unmatched_modes_position],
        ") for mode ", modes[unmatched_modes_position]
      ))
    }



    # Call C++ function for multiple matrix multiplication
    result_data <- ttm_multiple_cpp(tensor$data, matrix, modes, transpose)    
    result_tensor <- Tensor$new(result_data)
    
    # Apply squeeze to remove singleton dimensions for vector operations
    result_dims <- result_tensor$dim()
    new_dims <- result_dims[result_dims != 1]
    
    if (length(new_dims) == 0) {
      # All dimensions were 1, result is scalar
      return(Tensor$new(as.vector(result_tensor$data), c()))
    } else if (length(new_dims) < length(result_dims)) {
      # Some dimensions were 1, reshape to remove them
      return(result_tensor$reshape(new_dims))
    } else {
      # No singleton dimensions to remove
      return(result_tensor)
    }
  } else {
    stop("Input must be a matrix, vector, or list of matrices/vectors")
  }
}
