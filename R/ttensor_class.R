#' R6 Class for Tucker Tensors (TTensor)
#'
#' @description
#' A class representing Tucker tensors (TTensor).
#' A Tucker tensor is a core tensor multiplied by a matrix along each mode.
#'
#' @export
TTensor <- R6::R6Class("TTensor",
    public = list(
        #' @field core The core Tensor object
        core = NULL,

        #' @field U List of factor matrices
        U = NULL,

        #' @description
        #' Initialize a new TTensor
        #' @param core A Tensor object representing the core
        #' @param U A list of factor matrices
        #' @return A new TTensor object
        initialize = function(core = NULL, U = NULL) {
            if (is.null(core) && is.null(U)) {
                return(invisible(self))
            }

            if (!inherits(core, "Tensor")) stop("core must be a Tensor object.")

            if (!is.list(U)) stop("U must be a list of factor matrices.")
            for (i in seq_along(U)) {
                if (!is.matrix(U[[i]])) {
                    U[[i]] <- as.matrix(U[[i]])
                }
            }

            core_dims <- core$dim()
            if (length(U) != length(core_dims)) {
                stop("Number of factor matrices must equal the number of dimensions of the core tensor.")
            }

            for (i in seq_along(U)) {
                if (ncol(U[[i]]) != core_dims[i]) {
                    stop(paste("Factor matrix U[[", i, "]] must have", core_dims[i], "columns to match core dimension."))
                }
            }

            self$core <- core
            self$U <- U
            return(invisible(self))
        },

        #' @description
        #' Get dimensions of the tensor
        #' @return Integer vector of dimensions
        dim = function() {
            as.integer(sapply(self$U, nrow))
        },

        #' @description
        #' Get number of dimensions
        #' @return Integer number of dimensions
        ndims = function() {
            length(self$U)
        },

        #' @description
        #' Print the TTensor object
        #' @param ... Additional arguments
        #' @return Invisible self
        print = function(...) {
            cat("<TTensor object>\n")
            cat("Dimensions: ", paste(self$dim(), collapse = " x "), "\n")
            cat("Core tensor (core):\n")
            cat("  ", paste(self$core$dim(), collapse = " x "), " Dense Tensor\n")
            cat("Factor matrices (U): \n")
            for (i in seq_along(self$U)) {
                cat(" U[[", i, "]]: ", nrow(self$U[[i]]), " x ", ncol(self$U[[i]]), "\n", sep = "")
            }
            invisible(self)
        },

        #' @description
        #' Convert Tucker tensor to dense Tensor
        #' @return A new Tensor object representing the full dense tensor
        full = function() {
            res <- self$core$clone_tensor()
            # Pass list of matrices to ttm which is highly optimized for multiple factors
            # Wait, tensory::ttm handles lists!
            # We just need to give it all matrices and modes 1:N
            modes <- seq_along(self$U)
            # Tensory ttm handles lists optimally doing sequential contractions.
            res <- tensory::ttm(res, self$U, mode = modes, transpose = FALSE)
            return(res)
        }
    )
)

#' Create a Tucker tensor (TTensor)
#'
#' @param core A Tensor object representing the core
#' @param U A list of factor matrices
#' @return A TTensor object
#' @export
ttensor <- function(core = NULL, U = NULL) {
    res <- TTensor$new(core, U)
    class(res) <- c("TTensor", "Tensor", class(res))
    return(res)
}

#' S3 print method for TTensor
#' @param x A TTensor object
#' @param ... Additional arguments
#' @export
print.TTensor <- function(x, ...) {
    x$print(...)
}

#' S3 function to convert TTensor to full Tensor
#' @param x A TTensor object
#' @param ... Additional arguments
#' @return A dense Tensor object
#' @export
as.tensor.TTensor <- function(x, ...) {
    x$full()
}
