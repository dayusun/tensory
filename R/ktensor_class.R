#' R6 Class for Kruskal Tensors (KTensor)
#'
#' @description
#' A class representing Kruskal tensors (KTensor), which are the sum of outer products
#' of vectors. This matches the behavior of MATLAB's Tensor Toolbox ktensor.
#'
#' A Kruskal tensor is represented by a set of factor matrices and a vector of weights.
#'
#' @export
KTensor <- R6::R6Class("KTensor",
    public = list(
        #' @field lambda Vector of weights
        lambda = NULL,

        #' @field U List of factor matrices
        U = NULL,

        #' @description
        #' Initialize a new KTensor
        #' @param lambda A numeric vector of weights. If NULL, defaults to a vector of 1s.
        #' @param U A list of factor matrices
        #' @return A new KTensor object
        initialize = function(lambda = NULL, U = NULL) {
            if (is.null(lambda) && is.null(U)) {
                return(invisible(self))
            }

            if (!is.list(U)) stop("U must be a list of factor matrices.")
            for (i in seq_along(U)) {
                if (!is.matrix(U[[i]])) {
                    U[[i]] <- as.matrix(U[[i]])
                }
            }

            num_modes <- length(U)
            if (num_modes == 0) stop("U must contain at least one factor matrix.")

            R <- ncol(U[[1]])
            for (i in seq_along(U)) {
                if (ncol(U[[i]]) != R) stop("All factor matrices must have the same number of columns.")
            }

            if (is.null(lambda)) {
                lambda <- rep(1, R)
            } else {
                if (length(lambda) != R) stop("Length of lambda must match number of columns in factor matrices.")
            }

            self$lambda <- as.numeric(lambda)
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
        #' Print the KTensor object
        #' @param ... Additional arguments
        #' @return Invisible self
        print = function(...) {
            cat("<KTensor object>\n")
            cat("Dimensions: ", paste(self$dim(), collapse = " x "), "\n")
            cat("Weights (lambda): ", paste(round(self$lambda, 4), collapse = " "), "\n")
            cat("Factor matrices (U): \n")
            for (i in seq_along(self$U)) {
                cat(" U[[", i, "]]: ", nrow(self$U[[i]]), " x ", ncol(self$U[[i]]), "\n", sep = "")
            }
            invisible(self)
        },

        #' @description
        #' Convert Kruskal tensor to dense Tensor
        #' @return A new Tensor object representing the full dense tensor
        full = function() {
            dims <- self$dim()
            R <- length(self$lambda)

            # For a scalar result, honor the scalar-tensor convention (dims = integer(0))
            if (length(dims) == 0) {
                value <- if (length(self$lambda) == 0) 0 else sum(self$lambda)
                return(Tensor$new(as.double(value), integer(0), fast = TRUE))
            }

            N <- length(dims)
            if (R == 0L) {
                return(Tensor$new(array(0, dim = dims), dims = dims))
            }
            if (N == 1L) {
                data <- as.vector(as.matrix(self$U[[1]]) %*% self$lambda)
                return(Tensor$new(data, dims = dims))
            }

            # sum_r lambda_r u_1r o ... o u_Nr as one matrix product with a
            # Khatri-Rao product of the other factors. Using mode 1 (rows
            # fastest) or mode N (rows slowest) keeps the product in the
            # tensor's own column-major order, so no permutation is needed;
            # take the larger of the two so the Khatri-Rao factor
            # (prod(dims) / I_n rows) is the smaller one. A per-component
            # outer() loop instead allocated ~4 tensor-sized arrays per rank.
            n <- if (dims[N] > dims[1]) N else 1L
            A <- as.matrix(self$U[[n]])
            A <- A * rep(self$lambda, each = nrow(A))
            KR <- .kr_others(self$U, n)
            data <- if (n == 1L) tcrossprod(A, KR) else tcrossprod(KR, A)
            if (!is.double(data)) storage.mode(data) <- "double"
            dim(data) <- dims
            return(Tensor$new(data, dims = dims))
        }
    )
)

#' Create a Kruskal tensor (KTensor)
#'
#' @param lambda A numeric vector of weights, or list of matrices U if lambda is omitted.
#' @param U A list of factor matrices. Let NULL if passing list as `lambda`.
#' @return A KTensor object
#' @rdname KTensor
#' @export
ktensor <- function(lambda = NULL, U = NULL) {
    if (is.list(lambda) && is.null(U)) {
        U <- lambda
        lambda <- NULL
    }
    res <- KTensor$new(lambda, U)
    class(res) <- c("KTensor", "Tensor", class(res))
    return(res)
}

#' S3 print method for KTensor
#' @param x A KTensor object
#' @param ... Additional arguments
#' @export
print.KTensor <- function(x, ...) {
    x$print(...)
}

#' S3 function to convert KTensor to full Tensor
#' @param x A KTensor object
#' @param ... Additional arguments
#' @return A dense Tensor object
#' @export
as.tensor.KTensor <- function(x, ...) {
    x$full()
}
