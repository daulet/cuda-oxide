# cuBLAS GEMM

Demonstrates stream-aware dense linear algebra integration:

- regular row-major SGEMM through `cuda_core::Blas`;
- strided-batched row-major SGEMM through the same handle;
- a Rust-authored `#[kernel]` launched on the same stream after cuBLAS work.

Run with:

```bash
cargo oxide run cublas_gemm
```
