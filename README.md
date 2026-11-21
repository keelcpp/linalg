# keel::linalg

Dense linear algebra (BLAS Level 1/2/3, linear solvers, and matrix decompositions) for the [keel](https://github.com/keelcpp/keel) C++20 ecosystem — column-major layout, optional Google Highway SIMD acceleration + multi-threaded BLIS GEMM via `keel::ThreadPool`, zero STL headers in public API, `-fno-exceptions -fno-rtti` compatible.

## Features

- **BLAS Level 1**: `axpy` (`f64`/`c64`), `dot`, `dotc`, `dotu`, `nrm2`, `scal`.
- **BLAS Level 2**: `gemv` (standard and transposed), `ger`.
- **BLAS Level 3** (BLIS-blocked Highway SIMD + portable fallback): `gemm`, `trsm`, `syrk`, `lu_panel` for both real `f64` and complex `c64`.
- **Solvers & Decompositions**:
  - `lu_factor` — divide-and-conquer blocked LU with partial pivoting.
  - `la_solve` — general linear solve $A X = B$ (blocked LU for square, Householder QR for tall $m \ge n$).
  - `matrix_inv` — matrix inversion via blocked LU.
  - `cholesky_upper` — blocked upper-triangular Cholesky ($A = R^H R$).
  - `qr_householder` & `qr_pivoted` — Householder QR with optional column pivoting.
  - `svd_jacobi_inplace` — one-sided Jacobi thin SVD ($A = U \Sigma V^H$, $m \ge n$).

## Build & Test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```
