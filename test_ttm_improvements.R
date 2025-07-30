# Test script for TTM improvements
# This script tests the improved TTM implementation for correctness and performance

library(testthat)
library(microbenchmark)

# Source the package (assuming we're in the package directory)
source("R/tensor_class.R")
source("R/tensor_operations.R")

# Test helper functions
create_test_tensor <- function(dims) {
  data <- array(1:prod(dims), dim = dims)
  Tensor$new(data)
}

test_ttm_correctness <- function() {
  cat("Testing TTM correctness...\n")
  
  # Test 1: Basic 3D tensor with matrix multiplication
  cat("Test 1: Basic 3D tensor with matrix\n")
  t1 <- create_test_tensor(c(4, 3, 2))
  m1 <- matrix(1:8, nrow = 4, ncol = 2)
  
  result1 <- ttm(t1, m1, mode = 1)
  expected_dims1 <- c(2, 3, 2)  # 4->2, 3, 2
  
  if (!all(result1$dim() == expected_dims1)) {
    stop("Test 1 failed: Wrong dimensions")
  }
  cat("✓ Test 1 passed\n")
  
  # Test 2: Transpose functionality
  cat("Test 2: Transpose functionality\n")
  t2 <- create_test_tensor(c(3, 4, 2))
  m2 <- matrix(1:12, nrow = 4, ncol = 3)  # 4x3 matrix
  
  result2 <- ttm(t2, m2, mode = 2, transpose = TRUE)  # Contract with rows (4) of mode 2 (4)
  expected_dims2 <- c(3, 3, 2)  # 3, 4->3, 2
  
  if (!all(result2$dim() == expected_dims2)) {
    stop("Test 2 failed: Wrong dimensions for transpose")
  }
  cat("✓ Test 2 passed\n")
  
  # Test 3: Vector multiplication
  cat("Test 3: Vector multiplication\n")
  t3 <- create_test_tensor(c(4, 3, 2))
  v3 <- 1:4
  
  result3 <- ttm(t3, v3, mode = 1)
  expected_dims3 <- c(3, 2)  # Remove dimension 1 (4->scalar)
  
  if (!all(result3$dim() == expected_dims3)) {
    stop("Test 3 failed: Wrong dimensions for vector")
  }
  cat("✓ Test 3 passed\n")
  
  # Test 4: Multiple matrices
  cat("Test 4: Multiple matrices\n")
  t4 <- create_test_tensor(c(4, 3, 2))
  m4a <- matrix(1:8, nrow = 4, ncol = 2)
  m4b <- matrix(1:6, nrow = 3, ncol = 2)
  
  result4 <- ttm(t4, list(m4a, m4b), mode = c(1, 2))
  expected_dims4 <- c(2, 2, 2)  # 4->2, 3->2, 2
  
  if (!all(result4$dim() == expected_dims4)) {
    stop("Test 4 failed: Wrong dimensions for multiple matrices")
  }
  cat("✓ Test 4 passed\n")
  
  # Test 5: Multiple vectors
  cat("Test 5: Multiple vectors\n")
  t5 <- create_test_tensor(c(4, 3, 2))
  v5a <- 1:4
  v5b <- 1:3
  
  result5 <- ttm(t5, list(v5a, v5b), mode = c(1, 2))
  expected_dims5 <- c(2)  # Remove dimensions 1 and 2, keep 2
  
  if (!all(result5$dim() == expected_dims5)) {
    stop("Test 5 failed: Wrong dimensions for multiple vectors")
  }
  cat("✓ Test 5 passed\n")
  
  cat("All correctness tests passed! ✓\n\n")
}

test_error_handling <- function() {
  cat("Testing error handling...\n")
  
  t <- create_test_tensor(c(4, 3, 2))
  
  # Test invalid mode
  expect_error(ttm(t, matrix(1:6, 2, 3), mode = 0), "Mode must be between")
  expect_error(ttm(t, matrix(1:6, 2, 3), mode = 4), "Mode must be between")
  
  # Test dimension mismatch
  expect_error(ttm(t, matrix(1:6, 2, 3), mode = 1), "Matrix columns")
  expect_error(ttm(t, matrix(1:6, 3, 2), mode = 1, transpose = TRUE), "Matrix rows")
  
  # Test vector length mismatch
  expect_error(ttm(t, 1:3, mode = 1), "Vector length")
  
  # Test mixed types in list
  expect_error(ttm(t, list(matrix(1:8, 4, 2), 1:3), mode = c(1, 2)), 
               "All elements must be of the same type")
  
  cat("Error handling tests passed! ✓\n\n")
}

benchmark_performance <- function() {
  cat("Benchmarking performance...\n")
  
  # Create larger tensors for meaningful benchmarks
  big_tensor <- create_test_tensor(c(50, 40, 30))
  big_matrix <- matrix(rnorm(50 * 25), nrow = 50, ncol = 25)
  
  # Benchmark single operation
  single_bench <- microbenchmark(
    ttm_single = ttm(big_tensor, big_matrix, mode = 1),
    times = 10
  )
  
  cat("Single matrix multiplication benchmark:\n")
  print(single_bench)
  
  # Benchmark multiple operations
  matrices_list <- list(
    matrix(rnorm(50 * 25), nrow = 50, ncol = 25),
    matrix(rnorm(40 * 20), nrow = 40, ncol = 20),
    matrix(rnorm(30 * 15), nrow = 30, ncol = 15)
  )
  
  multiple_bench <- microbenchmark(
    ttm_multiple = ttm(big_tensor, matrices_list, mode = c(1, 2, 3)),
    times = 5
  )
  
  cat("\nMultiple matrix multiplication benchmark:\n")
  print(multiple_bench)
  
  cat("Performance benchmarking completed!\n\n")
}

memory_efficiency_test <- function() {
  cat("Testing memory efficiency...\n")
  
  # Create test data
  tensor_data <- create_test_tensor(c(100, 100, 50))
  matrix_data <- matrix(rnorm(100 * 50), nrow = 100, ncol = 50)
  
  # Monitor memory usage
  initial_memory <- gc()
  
  # Perform operations
  result1 <- ttm(tensor_data, matrix_data, mode = 1)
  result2 <- ttm(tensor_data, matrix_data, mode = 1, transpose = TRUE)
  
  # Check that results are valid
  if (is.null(result1) || is.null(result2)) {
    stop("Memory efficiency test failed: NULL results")
  }
  
  final_memory <- gc()
  
  cat("Memory usage test completed\n")
  cat("Initial memory usage:\n")
  print(initial_memory)
  cat("Final memory usage:\n")
  print(final_memory)
  
  cat("Memory efficiency test completed!\n\n")
}

# Run all tests
main <- function() {
  cat("=== TTM Improvements Test Suite ===\n\n")
  
  tryCatch({
    test_ttm_correctness()
    test_error_handling()
    benchmark_performance()
    memory_efficiency_test()
    
    cat("=== All Tests Completed Successfully! ===\n")
    
  }, error = function(e) {
    cat("Error during testing:", e$message, "\n")
    stop(e)
  })
}

# Execute if run directly
if (!interactive()) {
  main()
}
