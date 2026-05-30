# cublas-sys

Runtime-loaded bindings for the subset of NVIDIA cuBLAS used by cuda-oxide.

The crate resolves `libcublas.so` with `dlopen` at runtime, so building the
workspace does not require the CUDA Toolkit to be installed. It currently
exposes handle lifecycle, version probing, stream and math-mode binding,
`Sgemm`, and `SgemmStridedBatched`.
