# Capability Roadmap

This page tracks functionality gaps that become visible when cuda-oxide is
evaluated against large GPU workloads such as model-specific inference engines
with compressed attention, routing, and quantized expert execution.

The items below are intentionally framed as capabilities, not as commitments to
mirror any single CUDA API, library, or engine one-for-one. A future cuda-oxide
solution may expose the same functionality through a different abstraction if
that better fits the project.

## Large-Model Memory Residency Controls

Status: shipped. Some inference runtimes need more than ordinary device
allocations. They manage very large, long-lived model and KV-cache regions and
may choose among several memory-residency strategies depending on hardware,
system memory pressure, and latency goals.

cuda-oxide now provides a supported runtime path for workflows that need:

- `ManagedBuffer<T>` for managed allocations suitable for very large tensors or
  caches,
- `MappedHostBuffer<T>` for mapped host memory with a device-visible pointer,
- `RegisteredHostMemory<'a, T>` for pre-existing host memory registration,
- `MemoryAdvice`, `MemoryLocation`, and `StreamAttachment` for placement and
  access controls,
- asynchronous managed-memory prefetch through `ManagedBuffer::prefetch_to`,
- `ResidencyBuffer<T>` and `ResidencyRequest` as an application-level policy
  hook for choosing among residency strategies.

The goal is not to standardize one memory policy, but to let advanced Rust GPU
applications express and control the residency model they need without dropping
out of the cuda-oxide ecosystem for the entire feature. The
`memory_residency` example exercises the shipped path on a real CUDA kernel.

## Production Dense Linear Algebra Integration

Status: shipped. Large transformer workloads rely on high-throughput dense
linear algebra in multiple places: projection layers, prefill paths, output
projections, and batched attention substeps. Custom kernels alone are not
always the right building block for these paths.

cuda-oxide now provides a supported runtime path for workflows that need:

- `cuda_core::Blas` as a RAII cuBLAS handle tied to a `CudaContext`,
- row-major `f32` matrix multiplication through `Blas::sgemm`,
- row-major strided-batched `f32` matrix multiplication through
  `Blas::sgemm_strided_batched`,
- execution on a caller-provided `CudaStream`,
- compatibility with `DeviceBuffer<f32>` ownership and validation before
  entering cuBLAS,
- a way to combine optimized library GEMM with Rust-authored kernels in the
  same cuda-oxide launch flow.

The goal is capability coverage, not native reimplementation of every dense
linear algebra primitive. The `cublas_gemm` example exercises the shipped path
by running regular and strided-batched SGEMM, then launching a Rust kernel on
the same stream.

## Warp-Scoped Matrix Multiply Acceleration

Some important kernels naturally map to warp-scoped matrix multiply execution
rather than Hopper-style warpgroup execution or Blackwell datacenter tensor
memory flows. This matters for workloads that target Ampere-class hardware,
consumer Blackwell parts, or algorithmic shapes that fit warp-level MMA better
than larger asynchronous accelerator pipelines.

The missing functionality is accelerator coverage for kernels that need:

- warp-scoped matrix multiply execution,
- shared-memory tile staging and register-fragment style accumulation,
- low-precision inputs with wider accumulators,
- programming models suitable for small or medium tile shapes used inside
  larger inference kernels,
- portability across GPU targets where newer warpgroup or datacenter-only
  accelerator models are not the appropriate fit.

The goal is to cover this class of kernels functionally. It does not require
cuda-oxide to reproduce any one CUDA surface area exactly.

## Scalable Device-Side Selection and Sorting Primitives

Routing and sparse-attention workloads often need fast per-row selection over
hundreds or thousands of scores. A simple single-thread top-k loop is not enough
for these cases, and users may otherwise fall back to bespoke external kernels.

The missing functionality is a supported cuda-oxide path for scalable
selection-oriented GPU building blocks such as:

- per-row top-k selection over large score vectors,
- deterministic tie-breaking where model semantics require it,
- block- or warp-cooperative selection, sorting, or equivalent primitives,
- temporary-memory handling suitable for high-throughput kernels,
- enough flexibility to support router selection, indexer selection, and other
  sparse scheduling decisions without forcing each project to invent a new
  substrate.

This capability could be satisfied by native primitives, structured interop, or
another project-consistent design. The important roadmap target is the
functionality.

## First-Class Low-Precision Inference Data Types

Modern inference systems increasingly depend on data representations beyond
plain `f16` and `bf16`. Some workloads emulate narrower formats manually today,
but that leaves usability, validation, and interoperability uneven.

The missing functionality is a coherent low-precision type story for CUDA-facing
Rust GPU code, covering formats relevant to inference workloads, including:

- fp8-class representations,
- fp4- or MX-style representations where they are useful to accelerator paths,
- explicit conversion and packing semantics,
- predictable load/store behavior and comparison rules,
- interop with accelerator and math-library paths where practical.

The roadmap target is not to force all low-precision formats into core language
types. It is to give inference-oriented cuda-oxide programs a supported and
well-defined way to represent and move these values through real kernels.

## Notes on Scope

These items complement the existing supported-features matrix:

- some gaps already have partial escape hatches today,
- some are primarily host-runtime concerns,
- some are compiler or device-programming gaps,
- some may ultimately be better solved through interop than by reimplementing
  every lower-level CUDA concept inside cuda-oxide itself.

The common thread is that each capability materially affects whether a demanding
GPU application can stay within a Rust-first cuda-oxide architecture end to end.
