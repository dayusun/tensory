library(tensory)

test_that("In-place arithmetic operations work", {
  # Test addition
  t1 <- Tensor$new(1:6)
  result <- t1$add(2)
  expect_equal(as.vector(t1$as_array()), c(3, 4, 5, 6, 7, 8))
  expect_identical(result, t1)  # Should return self
  
  # Test subtraction
  t2 <- Tensor$new(1:6)
  result <- t2$subtract(1)
  expect_equal(as.vector(t2$as_array()), c(0, 1, 2, 3, 4, 5))
  expect_identical(result, t2)
  
  # Test multiplication
  t3 <- Tensor$new(1:4)
  result <- t3$multiply(2)
  expect_equal(as.vector(t3$as_array()), c(2, 4, 6, 8))
  expect_identical(result, t3)
  
  # Test division
  t4 <- Tensor$new(c(2, 4, 6, 8))
  result <- t4$divide(2)
  expect_equal(as.vector(t4$as_array()), c(1, 2, 3, 4))
  expect_identical(result, t4)
})

test_that("Tensor-tensor arithmetic operations work", {
  t1 <- Tensor$new(1:4)
  t2 <- Tensor$new(2:5)
  
  # Test addition
  result <- t1$add(t2)
  expect_equal(as.vector(t1$as_array()), c(3, 5, 7, 9))
  
  # Test subtraction
  t3 <- Tensor$new(10:13)
  t4 <- Tensor$new(1:4)
  result <- t3$subtract(t4)
  expect_equal(as.vector(t3$as_array()), c(9, 9, 9, 9))
  
  # Test multiplication
  t5 <- Tensor$new(1:4)
  t6 <- Tensor$new(2:5)
  result <- t5$multiply(t6)
  expect_equal(as.vector(t5$as_array()), c(2, 6, 12, 20))
  
  # Test division
  t7 <- Tensor$new(c(10, 20, 30, 40))
  t8 <- Tensor$new(c(2, 4, 5, 8))
  result <- t7$divide(t8)
  expect_equal(as.vector(t7$as_array()), c(5, 5, 6, 5))
})

test_that("Arithmetic operations handle dimension mismatches", {
  t1 <- Tensor$new(1:4)
  t2 <- Tensor$new(1:6)
  
  expect_error(t1$add(t2), "Tensors must have the same dimensions")
  expect_error(t1$subtract(t2), "Tensors must have the same dimensions")
  expect_error(t1$multiply(t2), "Tensors must have the same dimensions")
  expect_error(t1$divide(t2), "Tensors must have the same dimensions")
})

test_that("Operator syntax works", {
  # Test addition operator
  t1 <- tensor(1:4)
  t2 <- tensor(2:5)
  result <- t1 + t2
  expect_equal(as.vector(result$as_array()), c(3, 5, 7, 9))
  # Ensure original tensors are unchanged
  expect_equal(as.vector(t1$as_array()), 1:4)
  expect_equal(as.vector(t2$as_array()), 2:5)
  
  # Test subtraction operator
  result <- t2 - t1
  expect_equal(as.vector(result$as_array()), c(1, 1, 1, 1))
  
  # Test multiplication operator
  result <- t1 * t2
  expect_equal(as.vector(result$as_array()), c(2, 6, 12, 20))
  
  # Test division operator
  t3 <- tensor(c(10, 20, 30, 40))
  t4 <- tensor(c(2, 4, 5, 8))
  result <- t3 / t4
  expect_equal(as.vector(result$as_array()), c(5, 5, 6, 5))
})

test_that("Scalar operator syntax works", {
  # Test scalar-tensor operations
  t <- tensor(1:4)
  
  # Addition
  result <- t + 2
  expect_equal(as.vector(result$as_array()), c(3, 4, 5, 6))
  
  result <- 2 + t
  expect_equal(as.vector(result$as_array()), c(3, 4, 5, 6))
  
  # Subtraction
  result <- t - 1
  expect_equal(as.vector(result$as_array()), c(0, 1, 2, 3))
  
  result <- 10 - t
  expect_equal(as.vector(result$as_array()), c(9, 8, 7, 6))
  
  # Multiplication
  result <- t * 3
  expect_equal(as.vector(result$as_array()), c(3, 6, 9, 12))
  
  result <- 3 * t
  expect_equal(as.vector(result$as_array()), c(3, 6, 9, 12))
  
  # Division
  result <- t / 2
  expect_equal(as.vector(result$as_array()), c(0.5, 1, 1.5, 2))
  
  result <- 12 / t
  expect_equal(as.vector(result$as_array()), c(12, 6, 4, 3))
})

test_that("Unary minus works", {
  t <- tensor(1:4)
  result <- -t
  expect_equal(as.vector(result$as_array()), c(-1, -2, -3, -4))
  # Original should be unchanged
  expect_equal(as.vector(t$as_array()), 1:4)
})
