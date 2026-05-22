# cuda-core

Safe RAII wrappers around the CUDA driver API.

## Overview

This crate turns raw CUDA handles into Rust types with automatic cleanup on drop:

| Type                  | Wraps                  | Cleanup call                       |
|-----------------------|------------------------|------------------------------------|
| `CudaContext`         | `CUcontext` (primary)  | `cuDevicePrimaryCtxRelease_v2`     |
| `CudaStream`          | `CUstream`             | `cuStreamDestroy_v2`               |
| `CudaEvent`           | `CUevent`              | `cuEventDestroy_v2`                |
| `CudaModule`          | `CUmodule`             | `cuModuleUnload`                   |
| `CudaFunction`        | `CUfunction`           | (prevented from outliving module)  |
| `Blas`                | `cublasHandle_t`       | `cublasDestroy_v2`                 |
| `PinnedHostBuffer<T>` | pinned host memory     | `cuMemFreeHost`                    |
| `ManagedBuffer<T>`    | managed memory         | `cuMemFree_v2`                     |
| `MappedHostBuffer<T>` | mapped host memory     | `cuMemFreeHost`                    |
| `RegisteredHostMemory<'a, T>` | registered host slice | `cuMemHostUnregister`    |

## Key APIs

- **Context**: `CudaContext::new(ordinal)` retains the primary context, binds it to the calling thread, and returns an `Arc<CudaContext>`.
- **Streams**: `ctx.new_stream()` creates a non-blocking stream. `stream.launch_host_function(f)` enqueues an `FnOnce` callback after all prior stream work completes -- this is the bridge to Rust futures in `cuda-async`.
- **Modules**: `ctx.load_module_from_file("kernel.ptx")` / `ctx.load_module_from_ptx_src(src)` load compiled GPU code. `module.load_function("kernel_name")` extracts a callable function handle.
- **Memory**: Async (`malloc_async`, `free_async`, `memcpy_htod_async`, ...) and sync (`malloc_sync`, `free_sync`) device memory operations.
- **Device buffers**: `DeviceBuffer<T>` owns device memory and provides host-device transfer helpers for `T: DeviceCopy`.
- **BLAS**: `Blas::new(&ctx)` creates a cuBLAS handle tied to the cuda-oxide context. `sgemm` and `sgemm_strided_batched` enqueue row-major `f32` matrix multiplication on a caller-provided `CudaStream` and validate `DeviceBuffer` sizes before calling cuBLAS.
- **Pinned host memory**: `PinnedHostBuffer<T>` allocates page-locked host memory for faster transfers. The async transfer helpers (`DeviceBuffer::from_pinned_host`, `copy_from_pinned_host_async`, `copy_to_pinned_host_async`) are `unsafe` because they only enqueue the copy and the caller must keep the pinned buffer alive until `stream.synchronize()`. Use `copy_to_pinned_host` for a blocking DtoH helper that syncs internally.
- **Residency memory**: `ManagedBuffer<T>` owns CUDA managed memory and supports `MemoryAdvice`, `prefetch_to`, and stream attachment. `MappedHostBuffer<T>` owns page-locked host memory with a device-visible pointer. `RegisteredHostMemory<'a, T>` maps an existing mutable host slice for GPU access while tying the registration lifetime to the borrow.
- **Residency policies**: `ResidencyBuffer<T>::zeroed_with` and `from_slice_with` let callers choose `Managed` or `MappedHost` from a `ResidencyRequest` instead of hard-coding one allocation strategy.
- **Launch**: `launch_kernel(func, grid, block, smem, stream, params)` enqueues a kernel on a stream. `launch_kernel_ex(...)` adds cluster dimensions (Hopper+); `launch_kernel_cooperative(...)` enables `Grid::sync()` for grid-wide barriers.
- **Events**: `ctx.new_event(flags)` for synchronization points and GPU-side timing via `event.elapsed_ms(end)`.

## Usage

```rust
use cuda_core::{CudaContext, LaunchConfig};

let ctx = CudaContext::new(0)?;
let stream = ctx.new_stream()?;
let module = ctx.load_module_from_file("vecadd.ptx")?;
let func = module.load_function("vecadd")?;
// ... allocate memory, launch kernel, synchronize
```

```rust
use cuda_core::{Blas, SgemmConfig};

let blas = Blas::new(&ctx)?;
let mut config = SgemmConfig::new(m, n, k);
config.alpha = 1.0;
config.beta = 0.0;
blas.sgemm(&stream, config, &a_dev, &b_dev, &mut c_dev)?;
```
