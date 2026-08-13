# Export Tensor Data to a Text File

Writes a `Tensor`, matrix, `KTensor`, or `Sptensor` to a plain-text file
in the MATLAB Tensor Toolbox `exportdata` format, so files are
interchangeable with MATLAB's `importdata`/`exportdata` pair.

## Usage

``` r
export_data(x, fname, fmt = "%.16e")
```

## Arguments

- x:

  Object to export.

- fname:

  Path of the file to create.

- fmt:

  `sprintf` format used for numeric values (default `"%.16e"`).

## Value

Invisibly, `fname`.

## Details

Format by type (values in column-major / first-index-fastest order):

- `tensor`: `tensor`, order, sizes, one value per line.

- `matrix`: `matrix`, `2`, sizes, one row of values per line.

- `ktensor`: `ktensor`, order, sizes, rank, lambda line, then each
  factor as a `matrix` block.

- `sptensor`: `sptensor`, order, sizes, number of nonzeros, then one
  `i1 ... iN value` line per nonzero.

## See also

[`import_data()`](https://www.sundayu.me/tensory/reference/import_data.md)
