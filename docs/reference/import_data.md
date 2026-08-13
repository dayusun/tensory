# Import Tensor Data from a Text File

Reads a file written by
[`export_data()`](https://www.sundayu.me/tensory/reference/export_data.md)
(or MATLAB Tensor Toolbox `exportdata`) and reconstructs the
corresponding object.

## Usage

``` r
import_data(fname)
```

## Arguments

- fname:

  Path of the file to read.

## Value

A `Tensor`, matrix, `KTensor`, or `Sptensor` depending on the file
header.

## See also

[`export_data()`](https://www.sundayu.me/tensory/reference/export_data.md)
