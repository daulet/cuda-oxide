# Non-obvious lessons

- The `nvcr.io/nvidia/pytorch:26.04-py3` B300 pod image has CUDA 13.2 headers
  and GPU access, but no Rust or clang. Install clang with apt, then install the
  pinned Rust toolchain and add `rustfmt`/`clippy` serially before running Cargo;
  concurrent Cargo/rustup invocations can race while downloading components.
- CUDA 13.2 bindgen exposes managed-memory prefetch/advice as
  `cuMemPrefetchAsync_v2` / `cuMemAdvise_v2` with `CUmemLocation`, not the older
  integer-device APIs. Its `CUmemLocation` id is wrapped in an anonymous union,
  matching the layout workaround already used in `cuda-core` VMM code.
