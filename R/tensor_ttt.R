#' Tensor Times Tensor (ttt) Operation
#'
#' @description
#' Computes the generalized product (outer, inner, or contracted) of two tensors.
#'
#' @details
#' - If `dimsA` and `dimsB` are NULL, computes the outer product.
#' - If `dimsA` and `dimsB` are provided, computes the contracted product along the specified dimensions.
#' The sizes of the corresponding dimensions must match.
#'
#' @param tensorA A Tensor object
#' @param tensorB A Tensor object
#' @param dimsA Subscripts of the dimensions in `tensorA` to contract over.
#' @param dimsB Subscripts of the dimensions in `tensorB` to contract over. Defaults to `dimsA`.
#' @return A new Tensor object representing the product.
#'
#' @examples
#' t1 <- tensor(array(1:24, dim = c(4, 3, 2)))
#' t2 <- tensor(array(1:12, dim = c(3, 2, 2)))
#'
#' # Outer product
#' result_outer <- ttt(t1, t2)
#'
#' # Contract over dimensions
#' result_contracted <- ttt(t1, t2, dimsA = c(2, 3), dimsB = c(1, 2))
#'
#' @export
ttt <- function(tensorA, tensorB, dimsA = NULL, dimsB = dimsA) {
    if (inherits(tensorA, "KTensor")) tensorA <- as.tensor.KTensor(tensorA)
    if (inherits(tensorA, "TTensor")) tensorA <- as.tensor.TTensor(tensorA)
    if (inherits(tensorB, "KTensor")) tensorB <- as.tensor.KTensor(tensorB)
    if (inherits(tensorB, "TTensor")) tensorB <- as.tensor.TTensor(tensorB)

    if (!inherits(tensorA, "Tensor") || !inherits(tensorB, "Tensor")) stop("Both inputs to ttt must be Tensor objects")

    tensorA_data <- tensorA$data
    tensorB_data <- tensorB$data
    dimA_full <- tensorA$dims
    dimB_full <- tensorB$dims
    if (length(dimA_full) == 0) dimA_full <- length(tensorA_data)
    if (length(dimB_full) == 0) dimB_full <- length(tensorB_data)

    if (is.null(dimsA) && is.null(dimsB)) {
        # Outer product fast path
        res_dims <- as.integer(c(dimA_full, dimB_full))
        out_data <- as.double(outer(tensorA_data, tensorB_data))
        dim(out_data) <- res_dims
        return(Tensor$new(data = out_data, dims = res_dims, fast = TRUE))
    }

    if (is.null(dimsA) || is.null(dimsB)) stop("Both dimsA and dimsB must be specified for contracted product.")

    # Inner product fast path
    if (length(dimsA) == length(dimA_full) && length(dimsB) == length(dimB_full)) {
        if (length(dimA_full) != length(dimB_full) || any(dimA_full[dimsA] != dimB_full[dimsB])) {
            stop("Contracted dimension sizes do not match.")
        }
        return(Tensor$new(data = as.double(sum(tensorA_data * tensorB_data)), dims = integer(0), fast = TRUE))
    }

    # Partial Contraction Validation
    if (length(dimsA) != length(dimsB)) stop("dimsA and dimsB must have the same length")
    if (any(dimsA < 1) || any(dimsA > length(dimA_full))) stop("dimsA index out of bounds")
    if (any(dimsB < 1) || any(dimsB > length(dimB_full))) stop("dimsB index out of bounds")
    if (any(dimA_full[dimsA] != dimB_full[dimsB])) stop("Contracted dimension sizes do not match.")

    dimsA <- as.integer(dimsA)
    dimsB <- as.integer(dimsB)

    remA <- setdiff(seq_along(dimA_full), dimsA)
    remB <- setdiff(seq_along(dimB_full), dimsB)

    if (length(remA) > 0) {
        permA <- c(remA, dimsA)
        matA <- aperm(tensorA_data, permA)
        dim(matA) <- c(prod(dimA_full[remA]), prod(dimA_full[dimsA]))
    } else {
        matA <- array(tensorA_data, dim = c(1, prod(dimA_full[dimsA])))
    }

    if (length(remB) > 0) {
        permB <- c(dimsB, remB)
        matB <- aperm(tensorB_data, permB)
        dim(matB) <- c(prod(dimB_full[dimsB]), prod(dimB_full[remB]))
    } else {
        matB <- array(tensorB_data, dim = c(prod(dimB_full[dimsB]), 1))
    }

    # Matrix multiply invokes optimized Fortran BLAS `%*%` internally
    matC <- matA %*% matB

    out_dim <- c(dimA_full[remA], dimB_full[remB])
    if (length(out_dim) == 0) {
        # Match MATLAB behavior, returning a scalar Tensor
        return(Tensor$new(data = as.double(matC), dims = integer(0), fast = TRUE))
    }

    out_data <- as.double(matC)
    dim(out_data) <- out_dim

    return(Tensor$new(data = out_data, dims = out_dim, fast = TRUE))
}
