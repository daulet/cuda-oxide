# Roadmap execution board

## Active roadmap item

### 1. Large-Model Memory Residency Controls

- Status: complete
- Goal: let advanced cuda-oxide applications express the residency strategy for
  large tensors and cache regions without leaving the Rust runtime layer.
- Source surface: `crates/cuda-core`, related docs/examples/tests, and the
  supported-features matrix when the item is complete.
- Required evidence:
  - owned managed allocation support,
  - mapped host memory or an equivalent host-visible GPU-access surface,
  - registration of existing host memory for GPU access,
  - placement/advice controls,
  - asynchronous prefetch or staging of large regions,
  - an application-facing selection hook instead of one hard-coded policy.

#### Planned milestones

1. Runtime primitives and ownership model
   - Status: complete
   - End-state: `cuda-core` exposes the low-level driver wrappers and minimal
     public types needed to represent managed memory, mapped host memory, and
     registered host memory without leaking raw lifetime hazards.
   - Implementation plan:
     - add `cuMemAllocManaged`, mapped host allocation, host registration, and
       host device-pointer wrappers to `crates/cuda-core/src/memory.rs`;
     - add public RAII handles for owned managed memory, owned mapped host
       memory, and borrowed registered host memory;
     - cover empty, zero-sized, owned, and registered memory behavior with
       focused `cuda-core` tests.
   - Validation:
     - `cargo fmt --check` in reusable `default/cuda-oxide-b300` pod on
       `hou2-prod1`: passed.
     - `CUDA_HOME=/usr/local/cuda cargo test -p cuda-core -- --nocapture` in
       reusable `default/cuda-oxide-b300` B300 pod: passed; 3 unit tests, 7
       pinned-host tests, 7 residency tests, 2 VMM/P2P tests, and doctests.
     - Claude CLI non-interactive review: no blocking issues on second pass.

2. Residency controls and policy hook
   - Status: complete
   - End-state: callers can attach placement advice, prefetch managed regions
     asynchronously on a chosen stream, and select among residency strategies
     through a compact runtime-facing API rather than bespoke ad hoc calls.
   - Implementation plan:
     - add raw `cuMemAdvise`, `cuMemPrefetchAsync`, and
       `cuStreamAttachMemAsync` wrappers to `crates/cuda-core/src/memory.rs`;
     - expose typed managed-memory controls for placement/access advice,
       asynchronous prefetch, and stream attachment;
     - add a compact owned residency policy path that lets applications choose
       managed or mapped-host allocation from a request descriptor;
     - cover the control calls and policy allocation path with focused
       `cuda-core` tests.
   - Validation:
     - Local `cargo fmt --check`: passed.
     - `cargo fmt --check` in reusable `default/cuda-oxide-b300` pod on
       `hou2-prod1`: passed.
     - `CUDA_HOME=/usr/local/cuda cargo test -p cuda-core --test residency
       -- --nocapture` in the B300 pod: passed; 10 residency tests.
     - `CUDA_HOME=/usr/local/cuda cargo test -p cuda-core -- --nocapture` in
       the B300 pod: passed; 3 unit tests, 7 pinned-host tests, 10 residency
       tests, 2 VMM/P2P tests, and doctests.
     - Claude CLI non-interactive review: no blocking issues.

3. Example and hardware verification
   - Status: complete
   - End-state: an executable example demonstrates the residency controls on a
     CUDA path, and a B300 validation run proves the relevant runtime behavior.
   - Implementation plan:
     - add a `memory_residency` `cargo oxide run` example with a raw-pointer
       kernel so the existing typed launch macro does not need a broader
       slice-argument refactor;
     - exercise managed input/output advice, prefetch, and stream attachment,
       plus mapped host memory and registered existing host memory in the same
       kernel path;
     - verify the example on the reusable `default/cuda-oxide-b300` pod and
       capture the exact command output.
   - Validation:
     - `CUDA_HOME=/usr/local/cuda CUDA_OXIDE_LLC=/usr/bin/llc-21 cargo oxide
       doctor` in the reusable `default/cuda-oxide-b300` pod: passed.
     - `CUDA_HOME=/usr/local/cuda CUDA_OXIDE_LLC=/usr/bin/llc-21 cargo oxide
       run memory_residency` in the B300 pod: passed; auto-detected `sm_103`
       and printed `SUCCESS: memory residency kernel produced 1024 correct
       elements`.

4. Docs and roadmap closure
   - Status: complete
   - End-state: README/book/support matrix reflect the shipped capability and
     item 1 is marked complete on this board.
   - Implementation plan:
     - replace stale book text that says managed memory has no cuda-oxide
       wrapper;
     - add the residency API to the support matrix and `cuda-core` public docs;
     - update README highlights/examples and mark the roadmap item shipped.
   - Validation:
     - Local `cargo fmt --check`: passed.
     - Local `git diff --check`: passed.
     - `PATH=.venv/bin:$PATH make html` in `cuda-oxide-book`: passed.
     - Claude CLI non-interactive review: no blocking issues.

### 2. Production Dense Linear Algebra Integration

- Status: in-progress
- Goal: let cuda-oxide host programs call production dense linear algebra from
  the same stream, context, and buffer ownership model as Rust-authored kernels.
- Source surface: a runtime cuBLAS binding, `crates/cuda-core`, examples,
  tests, README/book/support matrix when the item is complete.
- Required evidence:
  - matrix multiplication for inference-style hot paths,
  - strided batched matrix multiplication,
  - stream-aware execution that composes with `CudaStream`,
  - `DeviceBuffer` ownership compatibility,
  - a runnable example combining optimized library math with cuda-oxide
    orchestration.

#### Planned milestones

1. Runtime cuBLAS binding and handle lifecycle
   - Status: complete
   - End-state: the workspace has a minimal runtime-loaded cuBLAS binding with
     typed status errors, version probing, handle creation/destruction, and
     stream binding primitives.
   - Implementation plan:
     - add a `cublas-sys` crate following the existing `libnvvm-sys` and
       `nvjitlink-sys` `dlopen` style;
     - resolve only the symbols needed for item 2: create/destroy, version,
       set-stream, `Sgemm`, and `SgemmStridedBatched`;
     - add focused tests that load CUDA 13.2 cuBLAS from the reusable B300 pod.
   - Validation:
     - `CUDA_HOME=/usr/local/cuda cargo fmt --check` in the reusable
       `default/cuda-oxide-b300` pod on `hou2-prod1`: passed.
     - `CUDA_HOME=/usr/local/cuda cargo check -p cublas-sys` in the B300 pod:
       passed.
     - `CUDA_HOME=/usr/local/cuda cargo test -p cublas-sys -- --nocapture` in
       the B300 pod: passed; 2 runtime tests loaded cuBLAS, created a handle,
       queried version, and bound the default stream.

2. Stream-aware `cuda-core` GEMM API
   - Status: complete
   - End-state: `cuda-core` exposes safe row-major `sgemm` and
     `sgemm_strided_batched` entry points that operate on `DeviceBuffer<f32>`
     and enqueue work on a caller-provided `CudaStream`.
   - Implementation plan:
     - add a `cuda_core::blas` module with a RAII cuBLAS handle bound to a
       `CudaContext`;
     - validate dimensions/strides against buffer lengths before calling
       cuBLAS;
     - implement row-major wrappers over cuBLAS column-major calls without a
       compatibility layer or alternate fallback kernel;
     - cover single and strided-batched GEMM with B300 tests against CPU
       reference results.
   - Validation:
     - Local `cargo fmt --check`: passed.
     - Local `git diff --check`: passed.
     - Local `cargo check -p cuda-core`: blocked by missing local CUDA headers
       at `/usr/local/cuda/include/cuda.h`; validation moved to the B300 pod.
     - `CUDA_HOME=/usr/local/cuda cargo fmt --check` in the reusable
       `default/cuda-oxide-b300` pod on `hou2-prod1`: passed.
     - `CUDA_HOME=/usr/local/cuda cargo test -p cublas-sys -- --nocapture` in
       the B300 pod: passed after adapting the sys handle ownership for
       `cuda-core`.
     - `CUDA_HOME=/usr/local/cuda cargo check -p cuda-core` in the B300 pod:
       passed.
     - `CUDA_HOME=/usr/local/cuda cargo test -p cuda-core --test blas
       -- --nocapture` in the B300 pod: passed; 1 BLAS integration test
       covering single GEMM, strided-batched GEMM, and validation.
     - `CUDA_HOME=/usr/local/cuda cargo test -p cuda-core -- --nocapture` in
       the B300 pod: passed; 3 unit tests, 1 BLAS test, 7 pinned-host tests,
       10 residency tests, 2 VMM/P2P tests, and doctests.
     - Claude CLI non-interactive review: no blocking issues.

3. Example and orchestration proof
   - Status: open
   - End-state: a `cargo oxide run` example demonstrates cuBLAS GEMM composed
     with cuda-oxide-managed buffers, streams, and a Rust-authored kernel.
   - Implementation plan:
     - add a dense linear algebra example that runs a custom Rust kernel before
       or after cuBLAS work on the same stream;
     - exercise both regular and strided-batched GEMM paths;
     - verify correctness on the reusable B300 pod.
   - Validation: reusable B300 pod in `hou2-prod1`, exact command/log capture.

4. Docs and roadmap closure
   - Status: open
   - End-state: README/book/support matrix describe the shipped dense linear
     algebra integration and item 2 is marked complete on this board.
   - Validation: doc/source consistency review plus reviewer gate before commit.

## Remaining roadmap items

### 3. Warp-Scoped Matrix Multiply Acceleration

- Status: open
- Plan: not started. Write a milestone plan before implementation begins.

### 4. Scalable Device-Side Selection and Sorting Primitives

- Status: open
- Plan: not started. Write a milestone plan before implementation begins.

### 5. First-Class Low-Precision Inference Data Types

- Status: open
- Plan: not started. Write a milestone plan before implementation begins.
