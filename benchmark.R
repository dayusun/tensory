## ---- include = FALSE---------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)


## ----basic-example, warning=FALSE, message=FALSE------------------------------
library(tensory)
library(rTensor)
library(bench)

# Create a random 3-mode tensor (100x100x100) and a matrix (50x100)
set.seed(123)
dims <- c(100, 100, 100)
data <- rnorm(prod(dims))

t_tensory <- tensory::Tensor$new(data = data, dim = dims)
t_rtensor <- rTensor::as.tensor(array(data, dim = dims))

mat <- matrix(rnorm(50 * 100), nrow = 50, ncol = 100)

# Benchmark Mode 1
bench::mark(
  tensory = tensory::ttm(t_tensory, mat, 1)$data,
  rTensor = rTensor::ttm(t_rtensor, mat, 1)@data,
  check = FALSE, iterations = 20
)

# Benchmark Mode 2
bench::mark(
  tensory = tensory::ttm(t_tensory, mat, 2)$data,
  rTensor = rTensor::ttm(t_rtensor, mat, 2)@data,
  check = FALSE, iterations = 20
)

# Benchmark Mode 3
bench::mark(
  tensory = tensory::ttm(t_tensory, mat, 3)$data,
  rTensor = rTensor::ttm(t_rtensor, mat, 3)@data,
  check = FALSE, iterations = 20
)


## ----ttt-benchmark, warning=FALSE, message=FALSE------------------------------
# Size of tensors
dimA <- c(30, 40, 50)
dimB <- c(50, 40, 20)
arrA <- array(rnorm(prod(dimA)), dim = dimA)
arrB <- array(rnorm(prod(dimB)), dim = dimB)

tA_tensory <- tensory::tensor(arrA)
tB_tensory <- tensory::tensor(arrB)

tA_rtensor <- rTensor::as.tensor(arrA)
tB_rtensor <- rTensor::as.tensor(arrB)

# Inner Product (Full Contraction)
arrInner <- array(rnorm(prod(dimA)), dim = dimA)
tA_inner <- tensory::tensor(arrInner)
tB_inner <- tensory::tensor(arrInner)
tA_rt_inner <- rTensor::as.tensor(arrInner)
tB_rt_inner <- rTensor::as.tensor(arrInner)

bench::mark(
  tensory = tensory::ttt(tA_inner, tB_inner, dimsA = 1:3)$data,
  rTensor = rTensor::innerProd(tA_rt_inner, tB_rt_inner),
  check = FALSE, iterations = 20
)

# Partial Contraction (dim 2 and 3 of A with dim 2 and 1 of B)
bench::mark(
  tensory = tensory::ttt(tA_tensory, tB_tensory, dimsA = c(2, 3), dimsB = c(2, 1))$data,
  rTensor = {
    A_unfold <- rTensor::unfold(tA_rtensor, row_idx=1, col_idx=c(2,3))
    B_unfold <- rTensor::unfold(tB_rtensor, row_idx=c(2,1), col_idx=3)
    C_mat <- A_unfold@data %*% B_unfold@data
    rTensor::as.tensor(array(C_mat, dim=c(dimA[1], dimB[3])))@data
  },
  check = FALSE, iterations = 20
)

# Outer Product
arrA_small <- array(rnorm(100), dim=c(10, 10))
arrB_small <- array(rnorm(100), dim=c(10, 10))
tA_ts_sm <- tensory::tensor(arrA_small)
tB_ts_sm <- tensory::tensor(arrB_small)
tA_rt_sm <- rTensor::as.tensor(arrA_small)
tB_rt_sm <- rTensor::as.tensor(arrB_small)

bench::mark(
  tensory = tensory::ttt(tA_ts_sm, tB_ts_sm)$data,
  rTensor = rTensor::as.tensor(outer(tA_rt_sm@data, tB_rt_sm@data))@data,
  check = FALSE, iterations = 20
)

