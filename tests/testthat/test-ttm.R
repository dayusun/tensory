library(tensory)

test_that("ttm single matrix multiplication works", {
  # Test 2D tensor (matrix) times matrix
  t2d <- tensor(matrix(1:12, nrow = 4, ncol = 3))
  m <- matrix(1:8, nrow = 2, ncol = 4)  # Now contract with columns (4 columns to match tensor mode 1)
  
  # Multiply in mode 1 (contract with matrix columns)
  result <- ttm(t2d, m, mode = 1)
  expected_dims <- c(2, 3)  # Should be (2, 3) since we multiplied 4x3 tensor by 2x4 matrix (contract 4 cols)
  expect_equal(dim(result$as_array()), expected_dims)
  
  # Multiply in mode 2 (contract with matrix columns)
  m2 <- matrix(1:9, nrow = 3, ncol = 3)  # 3 columns to match tensor mode 2 (size 3)
  result2 <- ttm(t2d, m2, mode = 2)
  expected_dims2 <- c(4, 3)  # Should be (4, 3) since we multiplied 4x3 tensor by 3x3 matrix (contract 3 cols)
  expect_equal(dim(result2$as_array()), expected_dims2)
})

test_that("ttm multiple matrices multiplication works", {
  # Test 3D tensor times multiple matrices
  t3d <- tensor(array(1:24, dim = c(4, 3, 2)))
  
  m1 <- matrix(1:8, nrow = 2, ncol = 4)  # For mode 1 - 4 columns to match tensor dim 1 (size 4)
  m2 <- matrix(1:6, nrow = 2, ncol = 3)  # For mode 2 - 3 columns to match tensor dim 2 (size 3)
  
  # Multiply with list of matrices
  result <- ttm(t3d, list(m1, m2), mode = c(1, 2))
  expected_dims <- c(2, 2, 2)  # Should be (2, 2, 2)
  expect_equal(dim(result$as_array()), expected_dims)
})

test_that("ttm transpose option works", {
  t2d <- tensor(matrix(1:12, nrow = 4, ncol = 3))
  m <- matrix(1:8, nrow = 4, ncol = 2)  # 4x2 matrix
  
  # With transpose = TRUE, effective matrix becomes 4x2
  # Contract tensor dim 0 (size 4) with matrix rows (size 4 after transpose)
  result <- ttm(t2d, m, mode = 1, transpose = TRUE)
  expected_dims <- c(2, 3)  # Should be (2, 3) - rows from matrix after transpose
  expect_equal(dim(result$as_array()), expected_dims)
})

test_that("ttm error handling works", {
  t2d <- tensor(matrix(1:12, nrow = 4, ncol = 3))
  m <- matrix(1:15, nrow = 5, ncol = 3)  # Wrong dimensions - need 5 columns to not match tensor
  
  # Should throw error for dimension mismatch
  expect_error(ttm(t2d, m, mode = 1), "Matrix columns.*must match tensor dimension")
  
  # Should throw error for invalid mode
  m2 <- matrix(1:8, nrow = 2, ncol = 4)
  expect_error(ttm(t2d, m2, mode = 3), "Mode must be between 1 and number of tensor dimensions")
})

test_that("ttm single vector multiplication works", {
  # Test 3D tensor times vector
  t3d <- tensor(array(1:24, dim = c(4, 3, 2)))
  v <- 1:4  # Vector of length 4 for mode 1
  
  result <- ttm(t3d, v, mode = 1)
  expected_dims <- c(3, 2)  # Should reduce by one dimension
  expect_equal(dim(result$as_array()), expected_dims)
  
  # Test with different mode
  v2 <- 1:3  # Vector of length 3 for mode 2
  result2 <- ttm(t3d, v2, mode = 2)
  expected_dims2 <- c(4, 2)  # Should reduce by one dimension
  expect_equal(dim(result2$as_array()), expected_dims2)
})

test_that("ttm multiple vectors multiplication works", {
  # Test 3D tensor times multiple vectors
  t3d <- tensor(array(1:24, dim = c(4, 3, 2)))
  v1 <- 1:4  # For mode 1
  v2 <- 1:3  # For mode 2
  
  # Multiply with list of vectors
  result <- ttm(t3d, list(v1, v2), mode = c(1, 2))
  expected_dims <- c(2)  # Should reduce to 1D
  expect_equal(dim(result$as_array()), expected_dims)
})

test_that("ttm vector error handling works", {
  t3d <- tensor(array(1:24, dim = c(4, 3, 2)))
  v <- 1:5  # Wrong length vector
  
  # Should throw error for dimension mismatch
  expect_error(ttm(t3d, v, mode = 1), "Vector length.*must match tensor dimension")
  
  # Should throw error for invalid mode
  v2 <- 1:4
  expect_error(ttm(t3d, v2, mode = 4), "Mode must be between 1 and number of tensor dimensions")
})

test_that("ttm mathematical correctness", {
  # Create a simple 2D tensor (matrix) - 2 rows, 3 columns
  t2d_data <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 2, ncol = 3, byrow = TRUE)
  t2d <- tensor(t2d_data)
  # t2d_data is:
  # [1, 3, 5]
  # [2, 4, 6]
  
  # Create a matrix to multiply with (3x2, so we contract 3 columns with tensor mode 1 which has size 2)
  # We need a matrix with 2 columns to contract with tensor rows (mode 1, size 2)
  m <- matrix(c(1, 2, 3, 4), nrow = 2, ncol = 2, byrow = TRUE)
  # m is:
  # [1, 3]
  # [2, 4]
  
  # Perform TTM on mode 1 (contract tensor rows with matrix columns)
  result <- ttm(t2d, m, mode = 1)
  
  # Manual calculation for column-based contraction:
  # For mode-1 multiplication, we contract tensor dimension 1 (size 2) with matrix columns (size 2)
  # Each column of result is M * corresponding column of tensor
  # But since we're contracting with columns, it's actually matrix multiplication of M with tensor rows
  # Result should be 2x3 (same column count, rows transformed by M)
  
  # Actually, let's think differently - mode-1 means we contract along the first dimension (rows)
  # So we contract the 2 rows of tensor with 2 columns of matrix
  # This gives us a result with dimensions: (2 columns from matrix) x (3 columns from tensor) = 2x3
  
  # Row 1 of result: [1, 3, 5] * M^T = [1*1 + 3*3 + 5*2, 1*2 + 3*4 + 5*4] = [1+9+10, 2+12+20] = [20, 34]
  # Row 2 of result: [2, 4, 6] * M^T = [2*1 + 4*3 + 6*2, 2*2 + 4*4 + 6*4] = [2+12+12, 4+16+24] = [26, 44]
  expected_result_data <- matrix(c(9, 12, 15, 19, 26, 33), nrow = 2, ncol = 3, byrow = TRUE)
  expected_result <- tensor(expected_result_data)
  
  expect_equal(result$as_array(), expected_result$as_array())
})

test_that("ttm transpose mathematical correctness", {
  # Create a simple 2D tensor (matrix)
  t2d_data <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 2, ncol = 3, byrow = TRUE)
  t2d <- tensor(t2d_data)
  
  # Create a matrix to multiply with (3x2)
  m <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 2, ncol = 3, byrow = TRUE)
  
  # Perform TTM with transpose
  result <- ttm(t2d, m, mode = 1, transpose = TRUE)
  
  # With transpose=TRUE, m becomes 3x2, so we contract 2 rows with tensor mode 1 (size 2)
  # This should work and give a 3x3 result
  expected_dims <- c(3, 3)
  expect_equal(dim(result$as_array()), expected_dims)
})
