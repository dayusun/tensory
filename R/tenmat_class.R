#' R6 Tenmat Class
#'
#' A class representing a matricized tensor, equivalent to MATLAB Tensor Toolbox's tenmat.
#'
#' @export
Tenmat <- R6::R6Class("Tenmat",
    public = list(
        #' @field data The underlying 2D matrix data
        data = NULL,

        #' @field rdims The modes mapped to the rows
        rdims = NULL,

        #' @field cdims The modes mapped to the columns
        cdims = NULL,

        #' @field tsize The size of the original tensor
        tsize = NULL,

        #' Initialize a new tenmat
        #'
        #' @param data The matrix representation
        #' @param rdims The row indices
        #' @param cdims The column indices
        #' @param tsize The size of the original tensor
        #' @return A new Tenmat object
        initialize = function(data = NULL, rdims = NULL, cdims = NULL, tsize = NULL) {
            if (is.null(data)) {
                self$tsize <- integer(0)
                self$rdims <- integer(0)
                self$cdims <- integer(0)
                self$data <- matrix(numeric(0), nrow = 0, ncol = 0)
                return(invisible(self))
            }

            # If we're cloning or direct instantiating with all internal fields available
            if (!is.null(rdims) && !is.null(cdims) && !is.null(tsize)) {
                self$tsize <- as.integer(tsize)
                self$rdims <- as.integer(rdims)
                self$cdims <- as.integer(cdims)

                if (!is.matrix(data)) {
                    data <- as.matrix(data)
                }
                self$data <- data
                class(self) <- c("Tenmat", class(self))
                return(invisible(self))
            }
            stop("Direct initialization requires data, rdims, cdims, and tsize. Use tenmat() constructor instead.")
        },

        #' Get dimension of the underlying matrix
        #' @return Integer vector of dimensions
        dim = function() {
            return(dim(self$data))
        },

        #' Get number of elements
        #' @return Integer number of elements
        length = function() {
            return(length(self$data))
        },

        #' Print tenmat information
        #'
        #' @return Invisible self
        print = function() {
            cat("<Tenmat object>\n")

            if (length(self$tsize) == 0) {
                cat("A matrix corresponding to a tensor of size [empty tensor]\n")
                cat("rindices = [ ] (modes of tensor corresponding to rows)\n")
                cat("cindices = [ ] (modes of tensor corresponding to columns)\n")
            } else {
                cat("A matrix corresponding to a tensor of size", paste(self$tsize, collapse = " x "), "\n")
                cat("rindices = [", paste(self$rdims, collapse = " "), "] (modes of tensor corresponding to rows)\n")
                cat("cindices = [", paste(self$cdims, collapse = " "), "] (modes of tensor corresponding to columns)\n")
            }
            cat("data =\n")
            print(self$data)
            invisible(self)
        },

        #' Show tenmat (alias for print)
        #'
        #' @return Invisible self
        show = function() {
            self$print()
        },

        #' Clone the tenmat
        #' @return A new Tenmat object
        clone_tenmat = function() {
            Tenmat$new(self$data, self$rdims, self$cdims, self$tsize)
        },

        #' Add another Tenmat
        #' @param other Another Tenmat object
        #' @return A new Tenmat object
        add = function(other) {
            if (!inherits(other, "Tenmat")) {
                stop("Argument must be a Tenmat object.")
            }
            if (!identical(self$tsize, other$tsize) ||
                !identical(self$rdims, other$rdims) ||
                !identical(self$cdims, other$cdims)) {
                stop("Tenmat objects must have identical original shapes and matrix alignments for addition.")
            }
            res <- self$clone_tenmat()
            res$data <- self$data + other$data
            return(res)
        },

        #' Subtract another Tenmat
        #' @param other Another Tenmat object
        #' @return A new Tenmat object
        subtract = function(other) {
            if (!inherits(other, "Tenmat")) {
                stop("Argument must be a Tenmat object.")
            }
            if (!identical(self$tsize, other$tsize) ||
                !identical(self$rdims, other$rdims) ||
                !identical(self$cdims, other$cdims)) {
                stop("Tenmat objects must have identical original shapes and matrix alignments for subtraction.")
            }
            res <- self$clone_tenmat()
            res$data <- self$data - other$data
            return(res)
        },

        #' Negate Tenmat
        #' @return A new Tenmat object
        negate = function() {
            res <- self$clone_tenmat()
            res$data <- -self$data
            return(res)
        },

        #' Transpose Tenmat
        #' @return A new Tenmat object
        transpose = function() {
            res <- self$clone_tenmat()
            res$rdims <- self$cdims
            res$cdims <- self$rdims
            res$data <- base::t(self$data)
            return(res)
        }
    )
)

#' Create a tenmat (tensor as matrix) object
#'
#' @param T A Tensor object, matrix, or Tenmat object
#' @param rdims Dimensions to map to rows
#' @param cdims Dimensions to map to columns (optional). Also accepts 't', 'fc', 'bc'.
#' @param tsize Tensor size (used internally)
#' @return A new Tenmat object
#' @export
tenmat <- function(T, rdims = NULL, cdims = NULL, tsize = NULL) {
    # Empty constructor
    if (missing(T)) {
        return(Tenmat$new())
    }

    # Copy Constructor
    if (inherits(T, "Tenmat")) {
        return(T$clone_tenmat())
    }

    # Matrix to Tenmat (when all 4 arguments are supplied directly)
    if (!missing(tsize) && (is.matrix(T) || (is.array(T) && length(dim(T)) == 2))) {
        n <- length(tsize)
        rdims <- as.integer(rdims)
        cdims <- as.integer(cdims)

        row_prod <- 1L
        if (length(rdims) > 0) row_prod <- as.integer(prod(tsize[rdims]))

        if (nrow(T) != row_prod) {
            stop("Number of rows of matrix must match size specified by rdims and tsize.")
        }
        if (ncol(T) != (length(T) %/% row_prod)) {
            stop("Number of columns of matrix must match size specified by cdims and tsize.")
        }

        return(Tenmat$new(T, rdims, cdims, tsize))
    }

    # If T is a standard R array/matrix, wrap it in a Tensor internally first
    # to handle conversions just like MATLAB's `tenmat(tensor(A))`
    if (is.numeric(T) && !inherits(T, "Tensor")) {
        T <- tensor(T)
    }

    if (!inherits(T, "Tensor")) {
        stop("First argument must be a Tensor, Tenmat, matrix, or empty.")
    }

    tsize_val <- T$dims
    n <- length(tsize_val)
    if (n == 0) tsize_val <- length(T$data) # For 1D edge cases

    # Handle the argument parsing logic
    if (is.null(cdims)) {
        rdims <- as.integer(rdims)
        cdims <- setdiff(1:n, rdims)
    } else if (is.character(cdims)) {
        if (cdims == "t") {
            # Transpose behavior
            actual_cdims <- as.integer(rdims)
            rdims <- setdiff(1:n, actual_cdims)
            cdims <- actual_cdims
        } else if (cdims == "fc") {
            # Forward cyclic
            rdims <- as.integer(rdims)
            if (length(rdims) != 1) stop("Only one row dimension allowed if option is 'fc'.")
            cdims <- c(
                if (n > rdims) seq(rdims + 1, n) else integer(0),
                if (rdims > 1) seq(1, rdims - 1) else integer(0)
            )
            # Cleanup any remaining edge cases via setdiff to ensure unique mapping
            cdims <- cdims[cdims <= n]
        } else if (cdims == "bc") {
            # Backward cyclic
            rdims <- as.integer(rdims)
            if (length(rdims) != 1) stop("Only one row dimension allowed if option is 'bc'.")

            front_seq <- integer(0)
            if (rdims > 1) front_seq <- (rdims - 1):1

            back_seq <- integer(0)
            if (n > rdims) back_seq <- n:(rdims + 1)

            cdims <- c(front_seq, back_seq)
        } else {
            stop("Unrecognized option string")
        }
    } else {
        rdims <- as.integer(rdims)
        cdims <- as.integer(cdims)
    }

    # Restructure the data using aperm to move the indices into place
    perm <- c(rdims, cdims)
    data_val <- aperm(T$data, perm)

    # Reshape efficiently
    row_size <- 1L
    if (length(rdims) > 0) row_size <- as.integer(prod(tsize_val[rdims]))

    dim(data_val) <- c(row_size, length(data_val) %/% row_size)

    return(Tenmat$new(data_val, rdims, cdims, tsize_val))
}
#' Convert object to Tenmat
#'
#' @param x Object to convert
#' @return A Tenmat object
#' @export
as.tenmat <- function(x) {
    if (inherits(x, "Tenmat")) {
        return(x)
    }
    if (inherits(x, "Tensor")) {
        return(tenmat(x, seq_along(x$dim())))
    }
    stop("Cannot convert object to Tenmat.")
}

#' Convert Tenmat to standard R Matrix
#'
#' @param x A Tenmat object
#' @param ... Additional arguments (ignored)
#' @return A matrix
#' @export
as.matrix.Tenmat <- function(x, ...) {
    return(x$data)
}

#' Convert Tenmat to standard R vector
#'
#' @param x A Tenmat object
#' @param mode The type of vector to be returned (e.g., "numeric", "character", "logical").
#' @return A vector
#' @export
as.vector.Tenmat <- function(x, mode = "any") {
    if (mode == "any") {
        return(as.vector(x$data))
    }
    return(as.vector(x$data, mode = mode))
}

#' Convert Tenmat to standard R double array (alias for matrix)
#'
#' @param x A Tenmat object
#' @return A double matrix
#' @export
double.Tenmat <- function(x) {
    return(x$data)
}

#' Convert Tenmat to standard R Matrix using generic type conversion
#'
#' @param x A Tenmat object
#' @param ... Additional arguments (ignored)
#' @return A double matrix
#' @export
as.double.Tenmat <- function(x, ...) {
    return(x$data)
}

#' Convert Tenmat to Tensor
#'
#' @param x A Tenmat object
#' @param ... Additional arguments (ignored)
#' @return A Tensor object
#' @export
as.tensor.Tenmat <- function(x, ...) {
    # Reverse the dimension mapping
    # The data is currently arranged as (rdims, cdims)

    if (length(x$tsize) == 0) {
        return(tensor(numeric(0), integer(0)))
    }

    # We need to reshape the flat data back to the (rdims, cdims) block structure
    ordered_dims <- c(x$tsize[x$rdims], x$tsize[x$cdims])
    arranged_data <- array(x$data, dim = ordered_dims)

    # The dimensions are currently ordered as c(rdims, cdims).
    # We need to permute them so they are back in order 1:ndims.
    # For aperm, perm[i] specifies which dimension of the original array goes into dimension i of the new array.
    # So if current layout is [2 1 3 4] (because rdims=2,1, cdims=3,4)
    # then we want to find the inverse permutation to get back to [1 2 3 4]
    current_order <- c(x$rdims, x$cdims)
    inverse_perm <- match(seq_along(x$tsize), current_order)

    original_data <- aperm(arranged_data, inverse_perm)

    return(tensor(original_data, x$tsize))
}

# Add S3 generics for the Tenmat class generic conversions for seamless interoperability
#' @export
as.tensor <- function(x, ...) UseMethod("as.tensor")

#' @export
as.tensor.default <- function(x, ...) {
    tensor(x)
}


# Basic operators for tenmat

#' @export
`+.Tenmat` <- function(e1, e2) {
    e1$add(e2)
}

#' @export
`-.Tenmat` <- function(e1, e2) {
    if (missing(e2)) {
        # Unary minus
        return(e1$negate())
    }
    e1$subtract(e2)
}

#' @export
t.Tenmat <- function(x) {
    x$transpose()
}

#' S3 Matrix Multiplication Generic
#'
#' @param x First argument
#' @param y Second argument
#' @return Result of matrix multiplication
#' @export
`%*%` <- function(x, y) {
    if (inherits(x, "Tenmat") || inherits(y, "Tenmat")) {
        return(`%*%.Tenmat`(x, y))
    }
    base::`%*%`(x, y)
}

#' @export
`%*%.Tenmat` <- function(x, y) {
    # Handle scalar input logic
    if (!inherits(y, "Tenmat") && length(y) == 1) {
        res <- x$clone_tenmat()
        res$data <- res$data * as.double(y)
        return(res)
    }

    if (!inherits(x, "Tenmat") && length(x) == 1) {
        res <- y$clone_tenmat()
        res$data <- res$data * as.double(x)
        return(res)
    }

    # Standardize both inputs to tenmats
    if (!inherits(x, "Tenmat")) x <- tenmat(x, 1)
    if (!inherits(y, "Tenmat")) y <- tenmat(y, 1)

    if (ncol(x$data) != nrow(y$data)) {
        stop("Size mismatch: Number of columns in A is not equal to the number of rows in B")
    }

    new_tsize <- c(x$tsize[x$rdims], y$tsize[y$cdims])

    if (length(new_tsize) > 0) {
        new_rdims <- seq_along(x$rdims)
        new_cdims <- seq_along(y$cdims) + length(x$rdims)
        new_data <- x$data %*% y$data

        return(Tenmat$new(new_data, new_rdims, new_cdims, new_tsize))
    } else {
        # If the combined scalar shapes are empty, fallback to scalar matrix multiply
        return(x$data %*% y$data)
    }
}
#' Matrix Multiplication Alias
#'
#' @param x First argument
#' @param y Second argument
#' @return Result of matrix multiplication
#' @export
mtimes <- function(x, y) {
    x %*% y
}
