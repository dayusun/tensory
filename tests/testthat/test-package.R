library(tensory)

test_that("Package loads correctly", {
  expect_true(requireNamespace("tensory", quietly = TRUE))
  expect_true("Tensor" %in% ls("package:tensory"))
  expect_true("tensor" %in% ls("package:tensory"))
  expect_true("zeros" %in% ls("package:tensory"))
  expect_true("ones" %in% ls("package:tensory"))
})

test_that("Basic functionality works after loading", {
  # Test that we can create tensors
  expect_s3_class(tensor(1:6), "Tensor")
  expect_s3_class(zeros(5), "Tensor")
  expect_s3_class(ones(c(2, 3)), "Tensor")
  
  # Test basic operations
  t1 <- tensor(1:4)
  t2 <- tensor(2:5)
  result <- t1 + t2
  expect_equal(as.vector(result$as_array()), c(3, 5, 7, 9))
})

test_that("S3 methods are properly exported", {
  expect_true(is.function(getS3method("+", "Tensor", optional = TRUE)))
  expect_true(is.function(getS3method("-", "Tensor", optional = TRUE)))
  expect_true(is.function(getS3method("*", "Tensor", optional = TRUE)))
  expect_true(is.function(getS3method("/", "Tensor", optional = TRUE)))
})
