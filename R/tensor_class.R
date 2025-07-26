
#' S4 `tensor` class
#'
#' @slot data An R  array.
#' @slot dims The dimension of the tensor.  
#'
#' @name tensor-class
#' @rdname tensor-class
#' @aliases tensor-class
#' @docType class
#' @export
setClass("tensor", slots = c(data = "ANY", dims = "integer"))

#' Create a `tensor` object
#' 
#' @param data A R vector, matrix or array.
#' @param dims The dimension of the tensor. When `dims` is not supplied, use the dimension of the data.
#' 
#' @return A \code{tensor} object.
#' @examples 
#' tensor(data=matrix(1:6,ncol=2),dims=c(2,3))
#' tensor(data=matrix(c(1L,2L,3L,4L),ncol=2) )
#' mytensor <- tensor(data=array(1:1000000,c(100,100,100)))
#' mytensor + mytensor
#' @export
tensor <- function(data, dims = NULL) {

      # Determine dimensions
      if (is.null(dims)) {
        dims <- if (is.vector(data)) length(data) else dim(data)
      }

      # Validate that dimensions match data length
      if (prod(dims) != length(data)) {
        stop("Product of specified dimensions must match the number of elements in data.")
      }

      # If `data` is not double, convert it (preserves matrix/array structure)
      if (!is.double(data)) data <- as.double(data)


      # Convert data to array and ensure dims is integer
      dims <- as.integer(dims)
      data <- array(data, dim = dims)



      # Instantiate a new tensor object
      new("tensor", data = data, dims = dims)
    }

#  validation function for tensor class object
setValidity("tensor", function(object) {
  if (length(object@data) != prod(object@dims)) {
    "Data length does not match the product of dimensions"
  } else {
    TRUE
  }
})
 
# show method for tensor class object
setMethod("show", "tensor", function(object) {
  cat("<tensor object>\n")
  cat("A tensor of order", length(object@dims), "with dimensions:", paste(object@dims, collapse = " x "), "\n")
  invisible(object)
})
 


# add method for tensor objects with `+` operator
setMethod("+", signature = c("tensor", "tensor"), function(e1, e2) {
  # if (!all(e1@dims == e2@dims)) {
  #   stop("Both tensors must have the same dimensions")
  # }
  new("tensor", data = e1@data + e2@data, dims = e1@dims)
})

## tensor add using test_add_cpp function
setGeneric("cpp_add", function(e1, e2) standardGeneric("cpp_add"))

setMethod("cpp_add", signature =  c(e1="tensor",e2="tensor") , function(e1, e2) {
  new("tensor", data =  test_add_cpp(e1@data, e2@data), dims = e1@dims)
})

setGeneric("cpp_add2", function(e1, e2) standardGeneric("cpp_add2"))
setMethod("cpp_add2", signature =  c(e1="tensor",e2="tensor") , function(e1, e2) {
  new("tensor", data =  tensor_add(e1@data, e2@data), dims = e1@dims)
})


# S7::method(print, tensor) <- function(x, ...) {
#     cat("<tensor object>\n")
#     cat("A tensor of order", length(x@dims), "with dimensions:", paste(x@dims, collapse = " x "), "\n")
#     
#     # Reshape the data into its array form and print it
#     # data_array <- array(x@data, dim = x@dims)
#     # print_truncated_array(data_array)
#     
#     invisible(x)  # Return x invisibly for chaining if needed
#   }
# 
# 
# S7::method(generic =`+`, signature = list(i = tensor, 
#                                           o = tensor)) <- function(e1, e2) {
#   if (!inherits(e2, "tensor") || !all(e1@dims == e2@dims)) {
#     stop("Both tensors must have the same dimensions")
#   }
#   tensor(e1@data + e2@data)
# }