library(tensory)

test_that("tensor() function works", {
  # Test basic usage
  t1 <- tensor(1:6)
  expect_s3_class(t1, "Tensor")
  expect_equal(as.vector(t1$as_array()), 1:6)
  
  # Test with explicit dimensions
  t2 <- tensor(1:6, c(2, 3))
  expect_equal(t2$dim(), c(2L, 3L))
  expect_equal(t2$as_array(), matrix(1:6, nrow = 2, ncol = 3))
  
  # Test with matrix input
  mat <- matrix(1:6, nrow = 2, ncol = 3)
  t3 <- tensor(mat)
  expect_equal(t3$as_array(), mat)
})

test_that("zeros() function works", {
  # Test 1D zeros
  t1 <- zeros(5)
  expect_equal(t1$dim(), 5L)
  expect_equal(as.vector(t1$as_array()), rep(0, 5))
  
  # Test 2D zeros
  t2 <- zeros(c(2, 3))
  expect_equal(t2$dim(), c(2L, 3L))
  expect_equal(t2$as_array(), matrix(0, nrow = 2, ncol = 3))
  
  # Test 3D zeros
  t3 <- zeros(c(2, 3, 4))
  expect_equal(t3$dim(), c(2L, 3L, 4L))
  expect_equal(t3$as_array(), array(0, dim = c(2, 3, 4)))
})

test_that("ones() function works", {
  # Test 1D ones
  t1 <- ones(5)
  expect_equal(t1$dim(), 5L)
  expect_equal(as.vector(t1$as_array()), rep(1, 5))
  
  # Test 2D ones
  t2 <- ones(c(2, 3))
  expect_equal(t2$dim(), c(2L, 3L))
  expect_equal(t2$as_array(), matrix(1, nrow = 2, ncol = 3))
  
  # Test 3D ones
  t3 <- ones(c(2, 3, 4))
  expect_equal(t3$dim(), c(2L, 3L, 4L))
  expect_equal(t3$as_array(), array(1, dim = c(2, 3, 4)))
})

test_that("Print and show methods work", {
  t <- tensor(1:6)
  
  # Test print method (capture output)
  output <- capture.output(t$print())
  expect_true(any(grepl("Tensor object", output)))
  expect_true(any(grepl("dimensions", output)))
  
  # Test show method
  output <- capture.output(t$show())
  expect_true(any(grepl("Tensor object", output)))
})

test_that("Dimension and length methods work", {
  t <- tensor(array(1:24, dim = c(2, 3, 4)))
  
  expect_equal(t$dim(), c(2L, 3L, 4L))
  expect_equal(t$length(), 24L)
  expect_equal(t$ndims(), 3L)
})

test_that("Sum method works", {
  # Test sum all elements
  t1 <- tensor(1:6)
  result1 <- t1$sum()
  expect_equal(as.vector(result1$as_array()), 21)
  expect_equal(result1$dim(), integer(0))
  
  # Test sum along specific dimensions (2D)
  mat <- matrix(1:6, nrow = 2, ncol = 3)
  t2 <- tensor(mat)
  
  # Sum along rows (dimension 1)
  result2 <- t2$sum(1)
  expected2 <- tensor(apply(mat, 2, sum))
  expect_equal(result2$as_array(), expected2$as_array())
  expect_equal(result2$dim(), c(3L))
  
  # Sum along columns (dimension 2)
  result3 <- t2$sum(2)
  expected3 <- tensor(apply(mat, 1, sum))
  expect_equal(result3$as_array(), expected3$as_array())
  expect_equal(result3$dim(), c(2L))
  
  # Test 3D sum
  arr3d <- array(1:24, dim = c(2, 3, 4))
  t3 <- tensor(arr3d)
  
  result4 <- t3$sum(1)
  expected4 <- tensor(apply(arr3d, c(2, 3), sum))
  expect_equal(result4$as_array(), expected4$as_array())
  expect_equal(result4$dim(), c(3L, 4L))

  result5 <- t3$sum(c(1, 3))
  expected5 <- tensor(apply(arr3d, 2, sum))
  expect_equal(result5$as_array(), expected5$as_array())
  expect_equal(result5$dim(), c(3L))
})
