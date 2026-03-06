#' R6 Tensor Class
#'
#' A modern tensor class for R that provides MATLAB Tensor Toolbox compatibility
#' with high-performance operations via xtensor C++ backend.
#'
#' @examples
#' # Create a tensor from a matrix
#' t <- Tensor$new(matrix(1:6, nrow = 2, ncol = 3))
#' print(t)
#'
#' # Create a tensor with specific dimensions
#' t2 <- Tensor$new(1:24, c(2, 3, 4))
#'
#' # Mathematical operations using method syntax
#' t3 <- t2$clone_tensor()$add(t2)
#'
#' # Mathematical operations using operator syntax
#' t4 <- t2 + t2
#' t5 <- t2 * 2
#' t6 <- 5 - t2
#' t7 <- t2 / 3
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
    #' @param fast If TRUE, bypasses all checks. Used internally for performance.
    #' @return A new Tensor object
    initialize = function(data = NULL, dims = NULL, fast = FALSE) {
      if (fast) {
        self$data <- data
        self$dims <- dims
        return(invisible(self))
      }

      if (is.null(data)) {
        return(invisible(self))
      }

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

      if (length(dims) == 0) {
        if (length(data) != 1) {
          stop("Product of specified dimensions must match the number of elements in data.")
        }
        self$data <- as.double(data)
        self$dims <- integer(0)
      } else if (length(dims) == 1 && dims == 1 && length(data) == 1) {
        # Handle 1D scalar case
        self$data <- array(as.double(data), dim = 1)
        self$dims <- 1L
      } else {
        # Validate that dimensions match data length
        if (prod(dims) != length(data)) {
          stop("Product of specified dimensions must match the number of elements in data.")
        }

        # Convert data to double array
        if (!is.double(data)) data <- as.double(data)
        self$data <- array(data, dim = dims)
        self$dims <- dims
      }
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

    #' Squeeze the tensor by removing singleton dimensions
    #' @return New Tensor object with singleton dimensions removed
    squeeze = function() {
      result_dims <- self$dims
      new_dims <- result_dims[result_dims != 1]

      if (length(new_dims) == 0) {
        # All dimensions were 1, result is scalar
        return(Tensor$new(as.vector(self$data), integer(0)))
      } else if (length(new_dims) < length(result_dims)) {
        # Some dimensions were 1, reshape to remove them
        return(Tensor$new(self$data, new_dims))
      } else {
        # No singleton dimensions to remove
        return(self$clone_tensor())
      }
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

    #' Element-wise modulo
    #' @param other Another Tensor or numeric value
    #' @return Self (in-place operation)
    modulo = function(other) {
      if (inherits(other, "Tensor")) {
        if (!identical(self$dims, other$dims)) {
          stop("Tensors must have the same dimensions for element-wise modulo")
        }
        self$data <- self$data %% other$data
      } else {
        self$data <- self$data %% as.double(other)
      }
      return(self)
    },

    #' Element-wise equality comparison
    #' @param other Another Tensor or numeric value
    #' @return Self (in-place operation)
    equal = function(other) {
      if (inherits(other, "Tensor")) {
        if (!identical(self$dims, other$dims)) {
          stop("Tensors must have the same dimensions for element-wise comparison")
        }
        self$data <- as.numeric(self$data == other$data)
      } else {
        self$data <- as.numeric(self$data == as.double(other))
      }
      return(self)
    },

    #' Element-wise inequality comparison
    #' @param other Another Tensor or numeric value
    #' @return Self (in-place operation)
    not_equal = function(other) {
      if (inherits(other, "Tensor")) {
        if (!identical(self$dims, other$dims)) {
          stop("Tensors must have the same dimensions for element-wise comparison")
        }
        self$data <- as.numeric(self$data != other$data)
      } else {
        self$data <- as.numeric(self$data != as.double(other))
      }
      return(self)
    },

    #' Element-wise less than comparison
    #' @param other Another Tensor or numeric value
    #' @return Self (in-place operation)
    less_than = function(other) {
      if (inherits(other, "Tensor")) {
        if (!identical(self$dims, other$dims)) {
          stop("Tensors must have the same dimensions for element-wise comparison")
        }
        self$data <- as.numeric(self$data < other$data)
      } else {
        self$data <- as.numeric(self$data < as.double(other))
      }
      return(self)
    },

    #' Element-wise less than or equal comparison
    #' @param other Another Tensor or numeric value
    #' @return Self (in-place operation)
    less_equal = function(other) {
      if (inherits(other, "Tensor")) {
        if (!identical(self$dims, other$dims)) {
          stop("Tensors must have the same dimensions for element-wise comparison")
        }
        self$data <- as.numeric(self$data <= other$data)
      } else {
        self$data <- as.numeric(self$data <= as.double(other))
      }
      return(self)
    },

    #' Element-wise greater than comparison
    #' @param other Another Tensor or numeric value
    #' @return Self (in-place operation)
    greater_than = function(other) {
      if (inherits(other, "Tensor")) {
        if (!identical(self$dims, other$dims)) {
          stop("Tensors must have the same dimensions for element-wise comparison")
        }
        self$data <- as.numeric(self$data > other$data)
      } else {
        self$data <- as.numeric(self$data > as.double(other))
      }
      return(self)
    },

    #' Element-wise greater than or equal comparison
    #' @param other Another Tensor or numeric value
    #' @return Self (in-place operation)
    greater_equal = function(other) {
      if (inherits(other, "Tensor")) {
        if (!identical(self$dims, other$dims)) {
          stop("Tensors must have the same dimensions for element-wise comparison")
        }
        self$data <- as.numeric(self$data >= other$data)
      } else {
        self$data <- as.numeric(self$data >= as.double(other))
      }
      return(self)
    },

    #' Element-wise logical NOT
    #' @return Self (in-place operation)
    logical_not = function() {
      self$data <- as.numeric(self$data == 0)
      return(self)
    },

    #' Element-wise logical AND
    #' @param other Another Tensor or numeric value
    #' @return Self (in-place operation)
    logical_and = function(other) {
      if (inherits(other, "Tensor")) {
        if (!identical(self$dims, other$dims)) {
          stop("Tensors must have the same dimensions for element-wise logical AND")
        }
        self$data <- as.numeric((self$data != 0) & (other$data != 0))
      } else {
        self$data <- as.numeric((self$data != 0) & (as.double(other) != 0))
      }
      return(self)
    },

    #' Element-wise logical OR
    #' @param other Another Tensor or numeric value
    #' @return Self (in-place operation)
    logical_or = function(other) {
      if (inherits(other, "Tensor")) {
        if (!identical(self$dims, other$dims)) {
          stop("Tensors must have the same dimensions for element-wise logical OR")
        }
        self$data <- as.numeric((self$data != 0) | (other$data != 0))
      } else {
        self$data <- as.numeric((self$data != 0) | (as.double(other) != 0))
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
  # If dims is provided and data is a scalar, create array filled with that scalar
  if (!is.null(dims) && is.numeric(data) && length(data) == 1) {
    dims <- as.integer(dims)
    total_elements <- prod(dims)
    data <- rep(data, total_elements)
  }
  Tensor$new(data, dims)
}

#' S3 print method for Tensor
#' @param x A Tensor object
#' @param ... Additional arguments
#' @export
print.Tensor <- function(x, ...) {
  x$print()
  invisible(x)
}

#' S3 show method for Tensor
#' @param object A Tensor object
#' @export
show.Tensor <- function(object) {
  object$show()
  invisible(object)
}

# S3 generics for arithmetic operations with Tensor objects
#' @export
`+.Tensor` <- function(e1, e2) {
  if (inherits(e1, "Tensor")) {
    return(e1$clone_tensor()$add(e2))
  } else {
    return(e2$clone_tensor()$add(e1))
  }
}

#' @export
`-.Tensor` <- function(e1, e2) {
  if (missing(e2)) {
    # Unary minus
    return(e1$clone_tensor()$multiply(-1))
  } else if (inherits(e1, "Tensor")) {
    return(e1$clone_tensor()$subtract(e2))
  } else {
    # e1 is scalar, e2 is Tensor
    # Create tensor with same dims as e2 filled with e1 value
    temp_tensor <- tensor(e1, e2$dim())
    return(temp_tensor$subtract(e2))
  }
}

#' @export
`*.Tensor` <- function(e1, e2) {
  if (inherits(e1, "Tensor")) {
    return(e1$clone_tensor()$multiply(e2))
  } else {
    return(e2$clone_tensor()$multiply(e1))
  }
}

#' @export
`/.Tensor` <- function(e1, e2) {
  if (inherits(e1, "Tensor")) {
    return(e1$clone_tensor()$divide(e2))
  } else {
    # e1 is scalar, e2 is Tensor
    # Create tensor with same dims as e2 filled with e1 value
    temp_tensor <- tensor(e1, e2$dim())
    return(temp_tensor$divide(e2))
  }
}

#' @export
`%%.Tensor` <- function(e1, e2) {
  if (inherits(e1, "Tensor")) {
    return(e1$clone_tensor()$modulo(e2))
  } else {
    # e1 is scalar, e2 is Tensor
    # Create tensor with same dims as e2 filled with e1 value
    temp_tensor <- tensor(e1, e2$dim())
    return(temp_tensor$modulo(e2))
  }
}

#' @export
`==.Tensor` <- function(e1, e2) {
  if (inherits(e1, "Tensor")) {
    return(e1$clone_tensor()$equal(e2))
  } else {
    # e1 is scalar, e2 is Tensor
    # Create tensor with same dims as e2 filled with e1 value
    temp_tensor <- tensor(e1, e2$dim())
    return(temp_tensor$equal(e2))
  }
}

#' @export
`!=.Tensor` <- function(e1, e2) {
  if (inherits(e1, "Tensor")) {
    return(e1$clone_tensor()$not_equal(e2))
  } else {
    # e1 is scalar, e2 is Tensor
    # Create tensor with same dims as e2 filled with e1 value
    temp_tensor <- tensor(e1, e2$dim())
    return(temp_tensor$not_equal(e2))
  }
}

#' @export
`<.Tensor` <- function(e1, e2) {
  if (inherits(e1, "Tensor")) {
    return(e1$clone_tensor()$less_than(e2))
  } else {
    # e1 is scalar, e2 is Tensor
    # Create tensor with same dims as e2 filled with e1 value
    temp_tensor <- tensor(e1, e2$dim())
    return(temp_tensor$less_than(e2))
  }
}

#' @export
`<=.Tensor` <- function(e1, e2) {
  if (inherits(e1, "Tensor")) {
    return(e1$clone_tensor()$less_equal(e2))
  } else {
    # e1 is scalar, e2 is Tensor
    # Create tensor with same dims as e2 filled with e1 value
    temp_tensor <- tensor(e1, e2$dim())
    return(temp_tensor$less_equal(e2))
  }
}

#' @export
`>.Tensor` <- function(e1, e2) {
  if (inherits(e1, "Tensor")) {
    return(e1$clone_tensor()$greater_than(e2))
  } else {
    # e1 is scalar, e2 is Tensor
    # Create tensor with same dims as e2 filled with e1 value
    temp_tensor <- tensor(e1, e2$dim())
    return(temp_tensor$greater_than(e2))
  }
}

#' @export
`>=.Tensor` <- function(e1, e2) {
  if (inherits(e1, "Tensor")) {
    return(e1$clone_tensor()$greater_equal(e2))
  } else {
    # e1 is scalar, e2 is Tensor
    # Create tensor with same dims as e2 filled with e1 value
    temp_tensor <- tensor(e1, e2$dim())
    return(temp_tensor$greater_equal(e2))
  }
}

#' @export
`!.Tensor` <- function(e1) {
  return(e1$clone_tensor()$logical_not())
}

#' @export
`&.Tensor` <- function(e1, e2) {
  if (inherits(e1, "Tensor")) {
    return(e1$clone_tensor()$logical_and(e2))
  } else {
    # e1 is scalar, e2 is Tensor
    # Create tensor with same dims as e2 filled with e1 value
    temp_tensor <- tensor(e1, e2$dim())
    return(temp_tensor$logical_and(e2))
  }
}

#' @export
`|.Tensor` <- function(e1, e2) {
  if (inherits(e1, "Tensor")) {
    return(e1$clone_tensor()$logical_or(e2))
  } else {
    # e1 is scalar, e2 is Tensor
    # Create tensor with same dims as e2 filled with e1 value
    temp_tensor <- tensor(e1, e2$dim())
    return(temp_tensor$logical_or(e2))
  }
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
