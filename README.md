# Sparse Matrix-Vector Multiply (CSR)

Implements `sparse_multiply`, which converts a dense row-major matrix into **Compressed Sparse Row (CSR)** format and computes `y = A * x` with zero dynamic memory allocation.

## Function Signature

```c
void sparse_multiply(
    int rows, int cols,   // Matrix dimensions
    const double* A,      // Dense input matrix (row-major)
    const double* x,      // Input vector (length: cols)
    int*    out_nnz,      // Output: number of non-zeros found
    double* values,       // Output CSR: non-zero values
    int*    col_indices,  // Output CSR: column index per non-zero
    int*    row_ptrs,     // Output CSR: row start pointers (length: rows+1)
    double* y             // Output: result vector (length: rows)
);
```

## How It Works

Scans `A` row by row. Non-zero values are written to `values[]` and their column indices to `col_indices[]`. Row start offsets are recorded in `row_ptrs[i]`, with `row_ptrs[rows]` as the closing sentinel. Then for each row, the dot product is accumulated using the CSR row extents and stored in `y[i]`.

## Complexity

- Time: `O(rows x cols)` to build CSR, `O(nnz)` for the product
- Extra space: `O(1)`, all buffers are caller-provided

## Build & Run

```bash
gcc -lm -o run challenge.c
./run
```
