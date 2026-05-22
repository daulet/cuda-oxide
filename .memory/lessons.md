# Non-obvious lessons

- The `nvcr.io/nvidia/pytorch:26.04-py3` B300 pod image has CUDA 13.2 headers
  and GPU access, but no Rust or clang. Install clang with apt, then install the
  pinned Rust toolchain and add `rustfmt`/`clippy` serially before running Cargo;
  concurrent Cargo/rustup invocations can race while downloading components.
- CUDA 13.2 bindgen exposes managed-memory prefetch/advice as
  `cuMemPrefetchAsync_v2` / `cuMemAdvise_v2` with `CUmemLocation`, not the older
  integer-device APIs. Its `CUmemLocation` id is wrapped in an anonymous union,
  matching the layout workaround already used in `cuda-core` VMM code.
- B300 `sm_103` module loading needs an LLVM `llc` new enough to emit matching
  PTX. LLVM 18 produced `.version 6.0` PTX and failed CUDA JIT with
  `DriverError(218, "a PTX JIT compilation failed")`; `llc-21` fixes the
  `cargo oxide run` path when passed through `CUDA_OXIDE_LLC=/usr/bin/llc-21`.
- CUDA 13.2 cuBLAS keeps `_v2` suffixes on create/destroy/version/set-stream
  and `cublasSgemm_v2`, but `cublasSgemmStridedBatched` is exported without a
  `_v2` suffix. Resolve that exact mixed symbol set when using `dlopen`.
- CUDA stream tests that return after an expected pre-call validation error
  still need to synchronize if earlier setup enqueued async allocation, copy, or
  memset work. Dropping `DeviceBuffer`s with that setup work still in flight can
  make later tests crash even though the validation path never called the GPU
  library under test.
- In `cuda-core` BLAS tests, separate `#[test]` cases for cuBLAS work crashed
  under the default Rust test harness even with a process-local mutex, while
  isolated tests and `--test-threads=1` passed. Keep related cuBLAS cases inside
  one test when they share one device/primary context.
- `kubectl cp local_dir pod:/existing/local_dir` nests the directory as
  `.../local_dir/local_dir`; copy individual files or copy nested contents back
  up before trusting pod validation.
- After changing a path dependency used by an example, `cargo oxide build` can
  keep using stale dependency metadata in the example `target/release` tree.
  Remove the specific generated `libcuda_device-*` and `.fingerprint` entries
  when the source file in the pod is correct but codegen still sees the old API.
- Rust 2024 rejects taking shared references to `static mut` shared-memory
  tiles. Use `&raw const TILE` and cast that raw pointer before passing row
  addresses to ldmatrix-style intrinsics.
