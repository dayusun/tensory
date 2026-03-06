#' Tensor Times Matrix/Vector (ttm) Operation
#'
#' @description
#' Compute a tensor times a matrix (or matrices) or vector (or vectors) in
#' one (or more) modes. This function implements the tensor times matrix/vector
#' operation similar to MATLAB's ttm/ttv functions. Uses efficient Xtensor-blas
#' implementation with optimized algorithms for high-performance tensor operations.
#'
#' @details
#' ## Mathematical Operations
#'
#' ### Single Matrix Multiplication
#' For a tensor \eqn{X \in \mathbb{R}^{I_1 \times I_2 \times \ldots \times I_n}} and matrix \eqn{M \in \mathbb{R}^{J \times I_k}}, the tensor times matrix
#' operation in mode k produces:
#'
#' \deqn{Y = X \times_k M}
#'
#' where \eqn{Y \in \mathbb{R}^{I_1 \times \ldots \times I_{k-1} \times J \times I_{k+1} \times \ldots \times I_n}}
#'
#' Mathematically: \deqn{Y_{i_1,\ldots,i_{k-1},j,i_{k+1},\ldots,i_n} = \sum_{i_k=1}^{I_k} X_{i_1,\ldots,i_k,\ldots,i_n} \times M_{j,i_k}}
#'
#' ### Single Vector Multiplication
#' For a tensor \eqn{X \in \mathbb{R}^{I_1 \times I_2 \times \ldots \times I_n}} and vector \eqn{v \in \mathbb{R}^{I_k}}, the tensor times vector
#' operation in mode k produces:
#'
#' \deqn{Y = X \times_k v}
#'
#' where \eqn{Y \in \mathbb{R}^{I_1 \times \ldots \times I_{k-1} \times I_{k+1} \times \ldots \times I_n}} (dimension reduced by 1)
#'
#' Mathematically: \deqn{Y_{i_1,\ldots,i_{k-1},i_{k+1},\ldots,i_n} = \sum_{i_k=1}^{I_k} X_{i_1,\ldots,i_k,\ldots,i_n} \times v_{i_k}}
#'
#' ### Multiple Matrices/Vectors
#' For multiple operations, the function applies contractions sequentially. For matrices \eqn{M_1, M_2, \ldots, M_m}
#' in modes \eqn{k_1, k_2, \ldots, k_m}:
#'
#' \deqn{Y = X \times_{k_1} M_1 \times_{k_2} M_2 \ldots \times_{k_m} M_m}
#'
#' ## Implementation Details
#'
#' The function uses an optimized algorithm for multiple matrix operations:
#'
#' 1. **Efficient Contraction Order**: Operations are sorted by mode in descending order to maintain
#'    computational efficiency during sequential tensor contractions.
#'
#' 2. **Single Final Transpose**: Instead of multiple intermediate transposes, all contractions are
#'    performed first, followed by a single transpose to restore natural axis ordering.
#'
#' 3. **Memory Optimization**: Direct tensordot operations minimize intermediate tensor copies and
#'    memory allocations.
#'
#' 4. **Automatic Dimension Management**: Singleton dimensions are automatically removed for vector
#'    operations, and axis permutations are calculated mathematically.
#'
#' ## Automatic Squeezing Behavior
#'
#' The `ttm` function automatically removes singleton dimensions (dimensions of size 1) from the result tensor:
#'
#' - **Vector operations**: Always squeeze singleton dimensions introduced by treating vectors as 1×n matrices
#' - **Matrix operations**: Generally preserve all dimensions, but may squeeze if matrix has size 1 in non-contracted dimension
#' - **Mixed operations**: Squeeze dimensions that become size 1 due to vector contractions
#' - **Scalar results**: When all dimensions are contracted or reduced to size 1, returns a scalar tensor with empty dimensions
#'
#' This behavior ensures that:
#' - Vector contractions properly reduce tensor dimensionality
#' - Results have minimal dimensionality without unnecessary singleton dimensions
#' - Mathematical operations produce expected tensor shapes
#'
#' ## Input Types
#'
#' - **Single matrix**: Contract tensor with one matrix along specified mode
#' - **Single vector**: Contract tensor with one vector, reducing dimensionality (automatically squeezed)
#' - **List of matrices**: Sequential contractions with multiple matrices
#' - **List of vectors**: Sequential contractions with multiple vectors (automatically squeezed)
#' - **Mixed list**: Combination of matrices and vectors (vectors treated as 1×n matrices, result squeezed)
#'
#' @param tensor A Tensor object
#' @param matrix A matrix, vector, or list of matrices/vectors to multiply with the tensor.
#'               For matrices: number of columns (or rows if transpose=TRUE) must match tensor dimension in specified mode.
#'               For vectors: length must match tensor dimension in specified mode.
#' @param mode Integer or vector specifying which mode(s) to multiply.
#'             For single matrix/vector, defaults to 1. For multiple matrices/vectors,
#'             must be a vector of modes corresponding to each matrix/vector.
#'             Negative modes specify the dimensions *not* to multiply.
#' @param transpose Logical. If TRUE, transpose matrices before multiplication.
#'                  Default is FALSE. Ignored for vectors.
#' @return A new Tensor object with the result of the multiplication.
#'         Singleton dimensions (size 1) are automatically removed (squeezed) from the result,
#'         particularly for vector operations which reduce tensor dimensionality.
#'         Scalar results have empty dimensions integer(0).
#'
#' @examples
#' # ===== SINGLE MATRIX EXAMPLES =====
#'
#' # Create a 3D tensor (4×3×2)
#' t3d <- tensor(array(1:24, dim = c(4, 3, 2)))
#' print(dim(t3d$as_array())) # [1] 4 3 2
#'
#' # Example 1: Single matrix in mode 1
#' # Matrix must have 4 columns to match tensor's first dimension
#' m1 <- matrix(1:8, nrow = 2, ncol = 4) # 2×4 matrix
#' result1 <- ttm(t3d, m1, mode = 1)
#' print(dim(result1$as_array())) # [1] 2 3 2 (4 replaced by 2)
#'
#' # Example 2: Single matrix in mode 2
#' # Matrix must have 3 columns to match tensor's second dimension
#' m2 <- matrix(1:9, nrow = 3, ncol = 3) # 3×3 matrix
#' result2 <- ttm(t3d, m2, mode = 2)
#' print(dim(result2$as_array())) # [1] 4 3 2 (3 replaced by 3)
#'
#' # Example 3: Matrix with transpose
#' # With transpose=TRUE, effective matrix is 4×2 (transposed from 2×4)
#' m3 <- matrix(1:8, nrow = 2, ncol = 4) # 2×4 matrix
#' result3 <- ttm(t3d, m3, mode = 1, transpose = TRUE)
#' print(dim(result3$as_array())) # [1] 2 3 2
#'
#' # ===== SINGLE VECTOR EXAMPLES =====
#'
#' # Example 4: Single vector (reduces dimensionality)
#' # Vector length must match tensor dimension in specified mode
#' v1 <- 1:4 # Length 4 vector for mode 1
#' result4 <- ttm(t3d, v1, mode = 1)
#' print(dim(result4$as_array())) # [1] 3 2 (dimension reduced: 4×3×2 → 3×2)
#'
#' # Example 5: Vector in different mode
#' v2 <- 1:3 # Length 3 vector for mode 2
#' result5 <- ttm(t3d, v2, mode = 2)
#' print(dim(result5$as_array())) # [1] 4 2 (dimension reduced: 4×3×2 → 4×2)
#'
#' # ===== MULTIPLE MATRICES EXAMPLES =====
#'
#' # Example 6: Two matrices in different modes
#' # First matrix: 2×4 for mode 1, second matrix: 3×3 for mode 2
#' m1 <- matrix(1:8, nrow = 2, ncol = 4) # 2×4 for mode 1
#' m2 <- matrix(1:9, nrow = 3, ncol = 3) # 3×3 for mode 2
#' result6 <- ttm(t3d, list(m1, m2), mode = c(1, 2))
#' print(dim(result6$as_array())) # [1] 2 3 2 (4→2, 3→3)
#'
#' # Example 7: Multiple matrices with all modes
#' # Contract all three modes with different matrices
#' m1 <- matrix(1:8, nrow = 2, ncol = 4) # 2×4 for mode 1
#' m2 <- matrix(1:6, nrow = 2, ncol = 3) # 2×3 for mode 2
#' m3 <- matrix(1:4, nrow = 2, ncol = 2) # 2×2 for mode 3
#' result7 <- ttm(t3d, list(m1, m2, m3), mode = c(1, 2, 3))
#' print(dim(result7$as_array())) # [1] 2 2 2
#'
#' # ===== MULTIPLE VECTORS EXAMPLES =====
#'
#' # Example 8: Two vectors (reduces dimensionality significantly)
#' v1 <- 1:4 # Length 4 for mode 1
#' v2 <- 1:3 # Length 3 for mode 2
#' result8 <- ttm(t3d, list(v1, v2), mode = c(1, 2))
#' print(dim(result8$as_array())) # [1] 2 (4×3×2 → 2, two dimensions reduced)
#'
#' # Example 9: All modes with vectors (results in scalar)
#' v1 <- 1:4 # Length 4 for mode 1
#' v2 <- 1:3 # Length 3 for mode 2
#' v3 <- 1:2 # Length 2 for mode 3
#' result9 <- ttm(t3d, list(v1, v2, v3), mode = c(1, 2, 3))
#' print(dim(result9$as_array())) # integer(0) - scalar result (squeezed)
#' print(result9$as_array()) # Single numeric value
#'
#' # ===== SQUEEZING BEHAVIOR EXAMPLES =====
#'
#' # Example 9b: Demonstrating automatic squeezing
#' t2d <- tensor(array(1:12, dim = c(4, 3))) # 2D tensor
#'
#' # Matrix that results in singleton dimension
#' m_singleton <- matrix(c(1, 1, 1, 1), nrow = 1, ncol = 4) # 1×4 matrix
#' result_singleton <- ttm(t2d, m_singleton, mode = 1)
#' print(dim(result_singleton$as_array())) # [1] 3 (singleton dimension removed)
#'
#' # Vector operation - automatic dimensionality reduction
#' v_reduce <- 1:4 # Length 4 vector
#' result_reduced <- ttm(t2d, v_reduce, mode = 1)
#' print(dim(result_reduced$as_array())) # [1] 3 (4×3 → 3, squeezed)
#'
#' # Multiple vector operations leading to scalar
#' v1_scalar <- 1:4 # For mode 1
#' v2_scalar <- 1:3 # For mode 2
#' result_scalar <- ttm(t2d, list(v1_scalar, v2_scalar), mode = c(1, 2))
#' print(dim(result_scalar$as_array())) # integer(0) - scalar (all dims squeezed)
#' print(as.numeric(result_scalar$as_array())) # The scalar value
#'
#' # ===== MIXED EXAMPLES =====
#'
#' # Example 10: Mix of matrix and vector
#' m1 <- matrix(1:8, nrow = 2, ncol = 4) # 2×4 matrix for mode 1
#' v2 <- 1:3 # Length 3 vector for mode 2
#' result10 <- ttm(t3d, list(m1, v2), mode = c(1, 2))
#' print(dim(result10$as_array())) # [1] 2 2 (matrix keeps dim, vector reduces)
#'
#' # ===== HIGHER-DIMENSIONAL EXAMPLES =====
#'
#' # Example 11: 4D tensor
#' t4d <- tensor(array(1:120, dim = c(5, 4, 3, 2)))
#' m1 <- matrix(1:15, nrow = 3, ncol = 5) # 3×5 for mode 1
#' m3 <- matrix(1:6, nrow = 2, ncol = 3) # 2×3 for mode 3
#' result11 <- ttm(t4d, list(m1, m3), mode = c(1, 3))
#' print(dim(result11$as_array())) # [1] 3 4 2 2 (5→3, 3→2)
#'
#' # ===== ERROR HANDLING EXAMPLES =====
#'
#' \dontrun{
#' # These will produce errors:
#'
#' # Wrong matrix dimensions
#' bad_matrix <- matrix(1:6, nrow = 2, ncol = 3) # 3 cols, but mode 1 needs 4
#' ttm(t3d, bad_matrix, mode = 1) # Error: dimension mismatch
#'
#' # Wrong vector length
#' bad_vector <- 1:5 # Length 5, but mode 1 needs length 4
#' ttm(t3d, bad_vector, mode = 1) # Error: dimension mismatch
#'
#' # Invalid mode
#' ttm(t3d, v1, mode = 4) # Error: mode out of bounds (tensor has 3 dimensions)
#'
#' # Mismatched modes and matrices count
#' ttm(t3d, list(m1, m2), mode = c(1)) # Error: length mismatch
#' }
#'
#' @export
ttm <- function(tensor, matrix, mode = NULL, transpose = FALSE) {
  if (!inherits(tensor, "Tensor")) {
    stop("ttm is not implemented for this object type")
  }
  UseMethod("ttm", matrix)
}

#' @export
ttm.matrix <- function(tensor, matrix, mode = NULL, transpose = FALSE) {
  tensor_dims <- tensor$dim()

  # Handle negative modes
  if (!is.null(mode) && all(mode < 0)) {
    mode <- setdiff(seq_along(tensor_dims), -mode)
    if (length(mode) == 0) {
      stop("All modes cannot be excluded")
    }
  } else if (!is.null(mode) && any(mode < 0)) {
    stop("Modes must be either all positive or all negative")
  }

  if (is.null(mode)) mode <- 1

  # Validate mode (ensure it's a single value)
  if (length(mode) != 1) {
    stop("Mode must be a single integer for single matrix operations")
  }

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
}

#' @export
ttm.array <- function(tensor, matrix, mode = NULL, transpose = FALSE) {
  if (length(dim(matrix)) != 2) {
    stop("ttm is not implemented for arrays with != 2 dimensions")
  }
  ttm.matrix(tensor, matrix, mode, transpose)
}

#' @export
ttm.numeric <- function(tensor, matrix, mode = NULL, transpose = FALSE) {
  tensor_dims <- tensor$dim()

  if (!is.null(mode) && all(mode < 0)) {
    mode <- setdiff(seq_along(tensor_dims), -mode)
    if (length(mode) == 0) stop("All modes cannot be excluded")
  } else if (!is.null(mode) && any(mode < 0)) {
    stop("Modes must be either all positive or all negative")
  }

  if (is.null(mode)) mode <- 1

  if (length(mode) != 1) {
    stop("Mode must be a single integer for single vector operations")
  }

  if (mode < 1 || mode > length(tensor_dims)) {
    stop("Mode must be between 1 and number of tensor dimensions")
  }

  tensor_mode_dim <- tensor_dims[mode]
  if (length(matrix) != tensor_mode_dim) {
    stop(paste0(
      "Vector length (", length(matrix), ") must match tensor dimension size (",
      tensor_mode_dim, ") for mode ", mode
    ))
  }

  if (storage.mode(matrix) != "double") {
    storage.mode(matrix) <- "double"
  }

  vector_matrix <- base::matrix(matrix, nrow = 1)
  result_tensor <- ttm.matrix(tensor, vector_matrix, mode = mode, transpose = FALSE)

  result_dims <- result_tensor$dim()
  if (length(result_dims) == 1 && result_dims[1] == 1) {
    return(Tensor$new(as.vector(result_tensor$data), integer(0)))
  } else {
    new_dims <- result_dims[result_dims != 1]
    if (length(new_dims) == 0) {
      return(Tensor$new(as.vector(result_tensor$data), integer(0)))
    } else {
      return(result_tensor$reshape(new_dims))
    }
  }
}

#' @export
ttm.integer <- function(tensor, matrix, mode = NULL, transpose = FALSE) {
  ttm.numeric(tensor, matrix, mode, transpose)
}

#' @export
ttm.logical <- function(tensor, matrix, mode = NULL, transpose = FALSE) {
  ttm.numeric(tensor, matrix, mode, transpose)
}

#' @export
ttm.list <- function(tensor, matrix, mode = NULL, transpose = FALSE) {
  tensor_dims <- tensor$dim()

  if (length(matrix) == 0) {
    stop("Empty list of matrices/vectors provided")
  }

  if (!is.null(mode) && all(mode < 0)) {
    mode <- setdiff(seq_along(tensor_dims), -mode)
    if (length(mode) == 0) stop("All modes cannot be excluded")
  } else if (!is.null(mode) && any(mode < 0)) {
    stop("Modes must be either all positive or all negative")
  }

  if (is.null(mode)) {
    modes <- seq_along(matrix)
  } else {
    modes <- mode
  }

  if (length(matrix) == length(tensor_dims) && length(modes) != length(tensor_dims)) {
    matrix <- matrix[modes]
  }

  num_matrices <- length(matrix)
  if (length(modes) != num_matrices) {
    stop(paste0("Number of modes (", length(modes), ") must match number of matrices/vectors (", num_matrices, ")"))
  }

  if (any(modes < 1) || any(modes > length(tensor_dims))) {
    stop("All modes must be between 1 and number of tensor dimensions")
  }

  element_types <- sapply(matrix, function(x) {
    if (is.matrix(x) || (is.array(x) && length(dim(x)) == 2)) {
      "matrix"
    } else if (is.vector(x) && !is.matrix(x) && !is.array(x)) {
      "vector"
    } else {
      "invalid"
    }
  })

  if (any(element_types == "invalid")) {
    invalid_indices <- which(element_types == "invalid")
    stop(paste0("Invalid elements at positions: ", paste(invalid_indices, collapse = ", "), ". All elements must be matrices/vectors."))
  }

  for (i in seq_along(matrix)) {
    if (element_types[i] == "vector") {
      matrix[[i]] <- base::matrix(matrix[[i]], nrow = 1)
    }
    if (storage.mode(matrix[[i]]) != "double") {
      storage.mode(matrix[[i]]) <- "double"
    }
  }

  if (transpose) {
    matrice_mode <- sapply(matrix, nrow)
  } else {
    matrice_mode <- sapply(matrix, ncol)
  }

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

  result_data <- ttm_multiple_cpp(tensor$data, matrix, modes, transpose)
  result_tensor <- Tensor$new(result_data)

  vector_modes <- modes[element_types == "vector"]
  if (length(vector_modes) > 0) {
    result_dims <- result_tensor$dim()
    keep_dims <- setdiff(seq_along(result_dims), vector_modes)
    new_dims <- result_dims[keep_dims]

    if (length(new_dims) == 0) {
      return(Tensor$new(as.vector(result_tensor$data), integer(0)))
    } else {
      return(result_tensor$reshape(new_dims))
    }
  } else {
    return(result_tensor)
  }
}

#' @export
ttm.default <- function(tensor, matrix, mode = NULL, transpose = FALSE) {
  stop("Input must be a matrix, vector, or list of matrices/vectors")
}
