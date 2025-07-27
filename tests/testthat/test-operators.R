library(tensory)

test_that("Modulo operator works", {
  # Test tensor-tensor modulo
  t1 <- tensor(c(10, 15, 20, 25))
  t2 <- tensor(c(3, 4, 6, 7))
  result <- t1 %% t2
  expect_equal(as.vector(result$as_array()), c(1, 3, 2, 4))
  # Ensure original tensors are unchanged
  expect_equal(as.vector(t1$as_array()), c(10, 15, 20, 25))
  expect_equal(as.vector(t2$as_array()), c(3, 4, 6, 7))
  
  # Test scalar-tensor modulo
  t3 <- tensor(c(10, 15, 20, 25))
  result <- t3 %% 3
  expect_equal(as.vector(result$as_array()), c(1, 0, 2, 1))
  
  # Test tensor-scalar modulo
  result <- 3 %% t3
  expect_equal(as.vector(result$as_array()), c(3, 3, 3, 3))
})

test_that("Comparison operators work", {
  # Test equality
  t1 <- tensor(c(1, 2, 3, 4))
  t2 <- tensor(c(1, 2, 5, 4))
  result <- t1 == t2
  expect_equal(as.vector(result$as_array()), c(1, 1, 0, 1))
  
  # Test inequality
  result <- t1 != t2
  expect_equal(as.vector(result$as_array()), c(0, 0, 1, 0))
  
  # Test less than
  result <- t1 < t2
  expect_equal(as.vector(result$as_array()), c(0, 0, 1, 0))
  
  # Test less than or equal
  result <- t1 <= t2
  expect_equal(as.vector(result$as_array()), c(1, 1, 1, 1))
  
  # Test greater than
  result <- t1 > t2
  expect_equal(as.vector(result$as_array()), c(0, 0, 0, 0))
  
  # Test greater than or equal
  result <- t1 >= t2
  expect_equal(as.vector(result$as_array()), c(1, 1, 0, 1))
})

test_that("Logical operators work", {
  # Test logical NOT
  t1 <- tensor(c(0, 1, 2, 0, 5))
  result <- !t1
  expect_equal(as.vector(result$as_array()), c(1, 0, 0, 1, 0))
  
  # Test logical AND
  t2 <- tensor(c(1, 0, 2, 0, 3))
  t3 <- tensor(c(2, 1, 0, 0, 4))
  result <- t2 & t3
  expect_equal(as.vector(result$as_array()), c(1, 0, 0, 0, 1))
  
  # Test logical OR
  result <- t2 | t3
  expect_equal(as.vector(result$as_array()), c(1, 1, 1, 0, 1))
})

test_that("In-place operations work for new operators", {
  # Test modulo
  t1 <- Tensor$new(c(10, 15, 20, 25))
  t2 <- Tensor$new(c(3, 4, 6, 7))
  result <- t1$modulo(t2)
  expect_equal(as.vector(t1$as_array()), c(1, 3, 2, 4))
  expect_identical(result, t1)
  
  # Test comparison operators
  t3 <- Tensor$new(c(1, 2, 3, 4))
  t4 <- Tensor$new(c(1, 2, 5, 4))
  result <- t3$equal(t4)
  expect_equal(as.vector(t3$as_array()), c(1, 1, 0, 1))
  expect_identical(result, t3)
  
  # Test logical operators
  t5 <- Tensor$new(c(0, 1, 2, 0, 5))
  result <- t5$logical_not()
  expect_equal(as.vector(t5$as_array()), c(1, 0, 0, 1, 0))
  expect_identical(result, t5)
})
