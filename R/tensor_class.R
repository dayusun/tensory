#' R6 Tensor Class
#'
#' A modern tensor class for R that provides MATLAB Tensor Toolbox compatibility
#' with high-performance operations via xtensor C++ backend.
#'
#' @examples
#' # Create a tensor from a matrix
#' t <- Tensor$new(matrix(1:6, nrow=2, ncol=3))
#' print(t)
#' 
#' # Create a tensor with specific dimensions
#' t2 <- Tensor$new(1:24, c(2, 3, 4))
#' 
#' # Mathematical operations
#' t3 <- t2$clone_tensor()$add(t2)
#' 
#' @export
Tensor <- R6::R6Class("Tensor",
  public = list(
    #' @field data The underlying array data
    data = NULL,
    
    #' @field dims The dimensions of the tensor
    dims = NULL,
    
    #' Initialize a new tensor
    #' 
    #' @param data A vector, matrix, array, or numeric value
    #' @param dims The dimensions of the tensor. If NULL, inferred from data
    #' @return A new Tensor object
    initialize = function(data, dims = NULL) {
      # Handle different input types
      if (is.null(dims)) {
        if (is.vector(data)) {
          dims <- length(data)
        } else if (is.matrix(data) || is.array(data)) {
          dims <- dim(data)
        } else if (is.numeric(data) && length(data) == 1) {
          dims <- 1
          data <- as.double(data)
        } else {
          stop("Unsupported data type")
        }
      }
      
      # Convert dims to integer
      dims <- as.integer(dims)
      
      # Handle scalar case
      if (length(dims) == 1 && dims == 1 && length(data) == 1) {
        data <- array(as.double(data), dim = 1)
      } else {
        # Validate that dimensions match data length
        if (prod(dims) != length(data)) {
          stop("Product of specified dimensions must match the number of elements in data.")
        }
        
        # Convert data to double array
        if (!is.double(data)) data <- as.double(data)
        data <- array(data, dim = dims)
      }
      
      self$data <- data
      self$dims <- dims
    },
    
    #' Get tensor dimensions
    #' @return Integer vector of dimensions
    dim = function() {
      return(self$dims)
    },
    
    #' Get number of elements
    #' @return Integer number of elements
    length = function() {
      return(length(self$data))
    },
    
    #' Get number of dimensions
    #' @return Integer number of dimensions
    ndims = function() {
      return(length(self$dims))
    },
    
    #' Convert to R array
    #' @return R array representation
    as_array = function() {
      return(self$data)
    },
    
    #' Print tensor information
    #' 
    #' @return Invisible self
    print = function() {
      cat("<Tensor object>\n")
      cat("A tensor of order", self$ndims(), "with dimensions:", paste(self$dims, collapse = " x "), "\n")
      invisible(self)
    },
    
    #' Show tensor (alias for print)
    #' 
    #' @return Invisible self
    show = function() {
      self$print()
    },
    
    #' Clone the tensor
    #' @return A new Tensor object with copied data
    clone_tensor = function() {
      Tensor$new(self$data)
    },
    
    #' Reshape the tensor
    #' @param new_dims New dimensions
    #' @return Self (in-place operation)
    reshape = function(new_dims) {
      new_dims <- as.integer(new_dims)
      if (prod(new_dims) != length(self$data)) {
        stop("Product of new dimensions must match the number of elements in tensor.")
      }
      self$data <- array(self$data, dim = new_dims)
      self$dims <- new_dims
      return(self)
    },
    
    #' Element-wise addition
    #' @param other Another Tensor or numeric value
    #' @return Self (in-place operation)
    add = function(other) {
      if (inherits(other, "Tensor")) {
        if (!identical(self$dims, other$dims)) {
          stop("Tensors must have the same dimensions for element-wise addition")
        }
        self$data <- self$data + other$data
      } else {
        self$data <- self$data + as.double(other)
      }
      return(self)
    },
    
    #' Element-wise subtraction
    #' @param other Another Tensor or numeric value
    #' @return Self (in-place operation)
    subtract = function(other) {
      if (inherits(other, "Tensor")) {
        if (!identical(self$dims, other$dims)) {
          stop("Tensors must have the same dimensions for element-wise subtraction")
        }
        self$data <- self$data - other$data
      } else {
        self$data <- self$data - as.double(other)
      }
      return(self)
    },
    
    #' Element-wise multiplication
    #' @param other Another Tensor or numeric value
    #' @return Self (in-place operation)
    multiply = function(other) {
      if (inherits(other, "Tensor")) {
        if (!identical(self$dims, other$dims)) {
          stop("Tensors must have the same dimensions for element-wise multiplication")
        }
        self$data <- self$data * other$data
      } else {
        self$data <- self$data * as.double(other)
      }
      return(self)
    },
    
    #' Element-wise division
    #' @param other Another Tensor or numeric value
    #' @return Self (in-place operation)
    divide = function(other) {
      if (inherits(other, "Tensor")) {
        if (!identical(self$dims, other$dims)) {
          stop("Tensors must have the same dimensions for element-wise division")
        }
        self$data <- self$data / other$data
      } else {
        self$data <- self$data / as.double(other)
      }
      return(self)
    },
    
    #' Sum along dimensions
    #' @param dims Dimensions to sum along (NULL for all)
    #' @return New Tensor with reduced dimensions
    sum = function(dims = NULL) {
      if (is.null(dims)) {
        result <- base::sum(self$data)
        return(Tensor$new(result, 1))
      } else {
        dims <- as.integer(dims)
        result_array <- base::apply(self$data, dims, base::sum)
        if (is.null(dim(result_array))) {
          result_array <- array(result_array, dim = length(result_array))
        }
        return(Tensor$new(result_array))
      }
    }
  )
)

#' Create a tensor object
#' 
#' Convenience function to create a Tensor object
#' 
#' @param data A vector, matrix, array, or numeric value
#' @param dims The dimensions of the tensor. If NULL, inferred from data
#' @return A new Tensor object
#' @export
tensor <- function(data, dims = NULL) {
  Tensor$new(data, dims)
}

#' Create a tensor of zeros
#' 
#' @param dims Dimensions of the tensor
#' @return A new Tensor object filled with zeros
#' @export
zeros <- function(dims) {
  dims <- as.integer(dims)
  data <- array(0.0, dim = dims)
  Tensor$new(data)
}

#' Create a tensor of ones
#' 
#' @param dims Dimensions of the tensor
#' @return A new Tensor object filled with ones
#' @export
ones <- function(dims) {
  dims <- as.integer(dims)
  data <- array(1.0, dim = dims)
  Tensor$new(data)
}
