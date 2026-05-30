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
     - Claude CLI non-interactive review: no blocking issues.
     - Claude CLI non-interactive review: no blocking issues.
     - Claude CLI non-interactive review: no blocking issues.

### DS4 Read-Only Registered Model Mapping Extension

- Status: complete
- Goal: expose the CUDA read-only mapped host registration contract needed by
  DS4 model-file ranges without weakening the immutable host borrow or hiding
  unsupported-device errors.
- Source surface: `crates/cuda-core` memory primitives, residency handles,
  tests, and public crate documentation.
- Required evidence:
  - a direct `DEVICEMAP | READ_ONLY` host-registration primitive,
  - an RAII read-only registered-host handle borrowing `&[T]`,
  - B300 runtime verification of either a valid device-visible mapping or the
    exact CUDA unsupported result that an application must handle,
  - no compatibility wrapper or implicit fallback in `cuda-core`.

#### Planned milestones

1. Read-only registered host range handle
   - Status: complete
   - End-state: `cuda-core` can register immutable host-backed model ranges as
     device-readable mapped memory with explicit CUDA error propagation.
   - Validation:
     - Local `cargo fmt --all -- --check`: passed.
     - Local `git diff --check`: passed.
     - B300 `cargo +nightly-2026-04-03 fmt --all -- --check`: could not run;
       the installed pod toolchain does not include the `rustfmt` component.
     - `CUDA_HOME=/usr/local/cuda-13.2 cargo +nightly-2026-04-03 test -p
       cuda-core --test residency -- --nocapture` in `default/ds4-rust-port-b300`
       on `hou2-prod1`: passed; 11 tests, with the live read-only registration
       branch returning `DriverError(801, "operation not supported")`.
     - `CUDA_HOME=/usr/local/cuda-13.2 cargo +nightly-2026-04-03 test -p
       cuda-core -- --nocapture` in the same B300 pod: passed; full touched
       crate suite including doctests.
     - Claude CLI non-interactive review: unavailable; `claude --bare -p`
       exited with `Not logged in`.

### DS4 Pageable Host HMM Mapping Extension

- Status: complete
- Goal: expose the CUDA pageable-memory access contract needed by DS4's
  read-only mmap prefetch branch without requiring applications to construct
  raw managed-memory locations or pass untracked host pointers to driver calls.
- Source surface: `crates/cuda-core` context capabilities, residency handles,
  tests, and public crate documentation.
- Required evidence:
  - a context query for pageable-memory device access capability,
  - a borrowed immutable pageable-host handle with advice and prefetch
    methods tied to the source lifetime,
  - B300 runtime verification of read-mostly advice and device prefetch on a
    page-aligned system allocation,
  - no DS4-specific policy or implicit fallback inside `cuda-core`.

#### Planned milestones

1. Read-only pageable host memory handle
   - Status: complete
   - End-state: `cuda-core` can safely express DS4's pageable HMM read path
     while returning `CUDA_ERROR_NOT_SUPPORTED` on devices lacking the
     required access capability.
   - Validation:
     - local `cargo fmt --all -- --check` and `git diff --check`: passed.
     - B300 `cargo +nightly-2026-04-03 test -p cuda-core --test residency
       -- --nocapture`: passed (12 tests); live output reported pageable access
       with `host_page_tables=false` and successful advice/prefetch while the
       read-only registered mapping still returned driver error `801`.
     - B300 `cargo +nightly-2026-04-03 test -p cuda-core -- --nocapture`:
       passed for the full touched crate suite.
     - Self-review: the safe API retains only an immutable host borrow and
       exposes synchronous advice; asynchronous prefetch was corrected to an
       `unsafe` method requiring the immutable backing range to live through
       stream completion, and device pointer use for GPU work remains within
       the existing unsafe CUDA call boundary.
     - Post-push DS4 integration review found the asynchronous-borrow API
       defect above before DS4 pinned the new handle; the corrected focused
       and full B300 `cuda-core` suites passed again.
     - Claude CLI non-interactive review: unavailable; `claude --bare -p`
       exited with `Not logged in`.

### 2. Production Dense Linear Algebra Integration

- Status: complete
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
   - Status: complete
   - End-state: a `cargo oxide run` example demonstrates cuBLAS GEMM composed
     with cuda-oxide-managed buffers, streams, and a Rust-authored kernel.
   - Implementation plan:
     - add a dense linear algebra example that runs a custom Rust kernel before
       or after cuBLAS work on the same stream;
     - exercise both regular and strided-batched GEMM paths;
     - verify correctness on the reusable B300 pod.
   - Validation:
     - Local `cargo fmt --check`: passed.
     - Local `cargo fmt --check --manifest-path
       crates/rustc-codegen-cuda/examples/cublas_gemm/Cargo.toml`: passed.
     - Local `git diff --check`: passed.
     - `CUDA_HOME=/usr/local/cuda CUDA_OXIDE_LLC=/usr/bin/llc-21 cargo oxide
       run cublas_gemm` in the reusable `default/cuda-oxide-b300` B300 pod on
       `hou2-prod1`: passed; auto-detected `sm_103` and printed
       `SUCCESS: cuBLAS GEMM paths matched CPU references`.
     - Claude CLI non-interactive review: no blocking issues.

4. Docs and roadmap closure
   - Status: complete
   - End-state: README/book/support matrix describe the shipped dense linear
     algebra integration and item 2 is marked complete on this board.
   - Implementation plan:
     - update the root README examples, status highlights, and crate map for
       `cublas_gemm`, `cuda_core::Blas`, and `cublas-sys`;
     - document the `Blas` surface in `crates/cuda-core/README.md` and the
       book API quick reference;
     - mark dense linear algebra support shipped in the supported-features
       matrix and capability roadmap.
   - Validation:
     - Local `cargo fmt --check`: passed.
     - Local `git diff --check`: passed.
     - `PATH=.venv/bin:$PATH make html` in `cuda-oxide-book`: passed.
     - Claude CLI non-interactive review: no blocking issues.

## Remaining roadmap items

### 3. Warp-Scoped Matrix Multiply Acceleration

- Status: complete
- Goal: let cuda-oxide kernels use warp-scoped tensor-core MMA for targets and
  tile shapes where WGMMA or tcgen05 are not the right fit.
- Source surface: `crates/cuda-device`, MIR importer/lowerer NVVM intrinsic
  plumbing, example kernels, tests, README/book/support matrix when complete.
- Required evidence:
  - warp-scoped `mma.sync` execution,
  - shared-memory tile staging through `ldmatrix`,
  - low-precision inputs with wider `f32` accumulators,
  - a programming model that can build small and medium GEMM tiles from repeated
    warp-level MMA steps,
  - validation on the reusable B300 pod.

#### Planned milestones

1. Warp MMA intrinsic surface and lowering
   - Status: complete
   - End-state: `cuda-device` exposes a compact `mma` module for the
     `m16n8k16` f16-input/f32-accumulator shape, and the compiler lowers its
     fragment loads and MMA call to real `ldmatrix` / `mma.sync` inline PTX.
   - Implementation plan:
     - add typed A, B, and accumulator fragment containers with explicit unsafe
       constructors for shared-memory tile loads;
     - add MIR importer and NVVM/LLVM lowering for `ldmatrix` loads and
       `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`;
     - add focused compile or PTX-shape checks that reject placeholder
       lowering and require the emitted instructions.
   - Validation:
     - Local `cargo fmt`: passed.
     - Local `cargo fmt --check`: passed.
     - Local `cargo fmt --check --manifest-path
       crates/rustc-codegen-cuda/examples/warp_mma_smoke/Cargo.toml`: passed.
     - Local `rustup run nightly cargo check -p cuda-device`: passed.
     - Local `rustup run nightly cargo check -p dialect-nvvm`: passed.
     - Local `rustup run nightly cargo check -p mir-lower`: passed.
     - Local `rustup run nightly cargo check -p mir-importer`: passed.
     - Local `git diff --check`: passed.
     - `CUDA_HOME=/usr/local/cuda cargo fmt --check` in the reusable
       `default/cuda-oxide-b300` pod on `hou2-prod1`: passed.
     - `CUDA_HOME=/usr/local/cuda cargo check -p mir-importer` in the B300 pod:
       passed.
     - `CUDA_HOME=/usr/local/cuda CUDA_OXIDE_LLC=/usr/bin/llc-21 cargo oxide
       build warp_mma_smoke --arch sm_80` in the B300 pod: passed.
     - Generated `warp_mma_smoke.ptx` contains
       `ldmatrix.sync.aligned.m8n8.x4.shared.b16`,
       `ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16`, and
       `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`.
     - `/usr/local/cuda/bin/ptxas --gpu-name sm_80 warp_mma_smoke.ptx -o
       /tmp/warp_mma_smoke.cubin` in the B300 pod: passed.
     - Claude CLI non-interactive review: no blocking issues.

2. Hardware-proven warp MMA example
   - Status: complete
   - End-state: a runnable example computes a reference-checked GEMM tile using
     shared-memory staging, repeated warp-level MMA, and register
     accumulation.
   - Implementation plan:
     - add a `warp_mma` example that stages f16 A/B tiles into shared memory,
       loads fragments with the new API, accumulates in `f32`, and stores the
       16x8 result tile;
     - run at least two K-tiles so the example proves repeated accumulation,
       not only a single instruction;
     - validate output numerically on the B300 pod and inspect generated PTX
       for `ldmatrix` and `mma.sync`.
   - Validation:
     - Local `cargo fmt --check`: passed.
     - Local `cargo fmt --check --manifest-path
       crates/rustc-codegen-cuda/examples/warp_mma/Cargo.toml`: passed.
     - Local `cargo fmt --check --manifest-path
       crates/rustc-codegen-cuda/examples/warp_mma_smoke/Cargo.toml`: passed.
     - Local `rustup run nightly cargo check -p cuda-device`: passed.
     - Local `rustup run nightly cargo check -p mir-importer`: passed.
     - Local `git diff --check`: passed.
     - `CUDA_HOME=/usr/local/cuda CUDA_OXIDE_LLC=/usr/bin/llc-21 cargo oxide
       run warp_mma` in the reusable `default/cuda-oxide-b300` B300 pod:
       passed; auto-detected `sm_103` and printed `SUCCESS: warp MMA tile
       matched CPU reference for 16x8x32; max error 0.000e0`.
     - `CUDA_HOME=/usr/local/cuda CUDA_OXIDE_LLC=/usr/bin/llc-21 cargo oxide
       build warp_mma_smoke --arch sm_80` in the B300 pod: passed after
       switching the smoke example to the same ldmatrix address pattern.
     - Generated `warp_mma.ptx` contains
       `ldmatrix.sync.aligned.m8n8.x4.shared.b16`,
       `ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16`, and
       `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`.
     - `/usr/local/cuda/bin/ptxas --gpu-name sm_103 warp_mma.ptx -o
       /tmp/warp_mma.cubin` in the B300 pod: passed.
     - Claude CLI non-interactive review: no blocking issues.

3. Docs and roadmap closure
   - Status: complete
   - End-state: README/book/support matrix describe warp-scoped MMA support and
     item 3 is marked complete on this board.
   - Validation:
     - Local `cargo fmt --check`: passed.
     - Local `git diff --check`: passed.
     - `PATH=.venv/bin:$PATH make html` in `cuda-oxide-book`: passed.

### 4. Scalable Device-Side Selection and Sorting Primitives

- Status: complete
- Goal: let cuda-oxide kernels perform deterministic cooperative top-k
  selection over rows large enough to need multiple lanes or a whole block.
- Source surface: `crates/cuda-device`, examples, README/book/support matrix
  when complete.
- Required evidence:
  - per-row top-k selection over row lengths larger than one warp,
  - deterministic tie-breaking,
  - block-cooperative execution,
  - explicit caller-provided temporary memory,
  - sorted top-k output suitable for routers, indexers, and sparse scheduling,
  - validation on the reusable B300 pod.

#### Planned milestones

1. Device selection value model and block primitive
   - Status: complete
   - End-state: `cuda-device` exposes a compact `selection` module with
     fixed-capacity top-k entries and a block-cooperative `f32` top-k primitive
     using caller-provided shared-memory scratch.
   - Implementation plan:
     - add `TopKEntry` and `TopK<K>` with higher-score-first ordering,
       deterministic lower-index tie-breaking, and NaN ranked last;
     - add `block_topk_f32<K, BLOCK_THREADS>` for one 1D block scanning one
       row by block-strided loads, merging per-thread top-k values through
       `SharedArray<TopK<K>, BLOCK_THREADS>` scratch;
     - cover compile-time constraints and basic API shape with local package
       checks before hardware validation.
   - Validation:
     - Local `cargo fmt --check`: passed.
     - Local `rustup run nightly cargo check -p cuda-device`: passed.
     - Local `rustup run nightly cargo test -p cuda-device selection --
       --nocapture`: passed; 2 selection ordering tests.
     - Local `git diff --check`: passed.
     - `CUDA_HOME=/usr/local/cuda cargo fmt --check` in the reusable
       `default/cuda-oxide-b300` pod on `hou2-prod1`: passed.
     - `CUDA_HOME=/usr/local/cuda cargo check -p cuda-device` in the B300 pod:
       passed.
     - Claude CLI non-interactive review: no blocking issues.

2. Hardware-validated top-k example
   - Status: complete
   - End-state: a runnable `topk_select` example computes top-k for multiple
     rows, including ties, and validates scores/indices against a CPU
     reference.
   - Implementation plan:
     - launch one block per row with a row length larger than the block size;
     - write sorted top-k `(score, index)` pairs from lane-local ranks;
     - validate all rows on the reusable B300 pod.
   - Validation:
     - Local `cargo fmt --check`: passed.
     - Local `cargo fmt --check --manifest-path
       crates/rustc-codegen-cuda/examples/topk_select/Cargo.toml`: passed.
     - Local `rustup run nightly cargo check -p cuda-device`: passed.
     - Local `rustup run nightly cargo test -p cuda-device selection --
       --nocapture`: passed; 2 selection ordering tests.
     - Local `git diff --check`: passed.
     - `CUDA_HOME=/usr/local/cuda CUDA_OXIDE_LLC=/usr/bin/llc-21 cargo oxide
       run topk_select` in the reusable `default/cuda-oxide-b300` B300 pod:
       passed; auto-detected `sm_103` and printed `SUCCESS: top-k selection
       matched CPU reference for 4 rows x 257 scores (K=4)`.
     - Claude CLI non-interactive review: no blocking issues.

3. Docs and roadmap closure
   - Status: complete
   - End-state: README/book/support matrix describe shipped selection support
     and item 4 is marked complete on this board.
   - Validation:
     - Local `cargo fmt --check`: passed.
     - Local `git diff --check`: passed.
     - `PATH=.venv/bin:$PATH make html` in `cuda-oxide-book`: passed.

### 5. First-Class Low-Precision Inference Data Types

- Status: complete
- Goal: give inference-oriented cuda-oxide programs first-class low-precision
  storage types with explicit conversion, packing, comparison, and
  host/device movement semantics.
- Source surface: a shared no-std low-precision type crate, `cuda-device`
  re-exports, `cuda-core` `DeviceCopy` integration, examples, README/book
  support matrix when complete.
- Required evidence:
  - fp8-class representations for common inference storage formats,
  - an fp4/MX-style packed representation useful for Blackwell-era data paths,
  - explicit `from_bits`/`to_bits`, `from_f32_sat`, `to_f32`, packing, and
    unpacking semantics,
  - deterministic comparison rules that do not hide NaN behavior,
  - `DeviceBuffer` compatibility for host/device transfers,
  - at least one real kernel moving and converting these values on B300.

#### Planned milestones

1. Shared low-precision value model
   - Status: complete
   - End-state: the workspace has a no-std low-precision crate that can be used
     from both host and device code, with compact `repr(transparent)` storage
     types and exhaustive bit-level tests.
   - Implementation plan:
     - add a workspace `cuda-lowp` crate for `Fp8E4M3`, `Fp8E5M2`,
       `Fp4E2M1`, and packed helpers such as fp8 pairs and fp4 pairs;
     - define exact bit layouts, canonical NaN handling, saturating finite
       conversion from `f32`, widening to `f32`, and deterministic total
       comparison methods;
     - re-export the types from `cuda-device` without adding host-runtime
       dependencies to the device crate;
     - verify the chosen CUDA/NVIDIA format names and encodings against the
       CUDA 13.2 headers in the reusable B300 pod before locking the API.
   - Validation:
     - CUDA 13.2 header/probe check in the reusable `default/cuda-oxide-b300`
       B300 pod: confirmed `__NV_E4M3`, `__NV_E5M2`, `__NV_E2M1`,
       round-to-nearest-even fp8/fp4 conversion behavior, fp4 NaN to positive
       maxnorm, and low-lane-first fp8/fp4 packing order.
     - Local `cargo fmt --check`: passed.
     - Local `cargo test -p cuda-lowp -- --nocapture`: passed; 8 unit tests.
     - Local `rustup run nightly cargo check -p cuda-lowp`: passed.
     - Local `rustup run nightly cargo check -p cuda-device`: passed.
     - Local `git diff --check`: passed.
     - `CUDA_HOME=/usr/local/cuda cargo fmt --check` in the reusable B300 pod:
       passed.
     - `CUDA_HOME=/usr/local/cuda cargo test -p cuda-lowp -- --nocapture` in
       the B300 pod: passed; 8 unit tests.
     - `CUDA_HOME=/usr/local/cuda cargo check -p cuda-device` in the B300 pod:
       passed.
     - Claude CLI non-interactive review: no blocking issues.

2. Host runtime movement and device conversion proof
   - Status: complete
   - End-state: low-precision values can be stored in `DeviceBuffer`s,
     transferred between host and device as typed values, and converted inside
     a Rust-authored kernel.
   - Implementation plan:
     - add `cuda-core::DeviceCopy` impls for the low-precision storage types;
     - add a `lowp_roundtrip` `cargo oxide run` example that quantizes `f32`
       inputs to fp8/fp4 forms, stores packed values, reloads them in a kernel,
       widens them, and checks host/device agreement;
     - cover edge cases explicitly: signed zeros, saturation, infinities, NaN,
       tie rounding, and packed nibble/byte ordering.
   - Validation:
     - Local `cargo fmt --check`: passed.
     - Local `cargo fmt --check --manifest-path
       crates/rustc-codegen-cuda/examples/lowp_roundtrip/Cargo.toml`: passed.
     - Local `cargo test -p cuda-lowp -- --nocapture`: passed; 8 unit tests.
     - Local `rustup run nightly cargo check -p cuda-device`: passed.
     - Local `git diff --check`: passed.
     - Local `rustup run nightly cargo check -p cuda-core`: blocked by missing
       local CUDA headers at `/usr/local/cuda/include/cuda.h`; validation moved
       to the B300 pod.
     - `CUDA_HOME=/usr/local/cuda cargo fmt --check` in the reusable B300 pod:
       passed.
     - `CUDA_HOME=/usr/local/cuda cargo test -p cuda-lowp -- --nocapture` in
       the B300 pod: passed; 8 unit tests.
     - `CUDA_HOME=/usr/local/cuda cargo check -p cuda-core` in the B300 pod:
       passed.
     - `CUDA_HOME=/usr/local/cuda cargo check -p cuda-device` in the B300 pod:
       passed.
     - `cargo fmt --check --manifest-path
       crates/rustc-codegen-cuda/examples/lowp_roundtrip/Cargo.toml` in the
       B300 pod: passed.
     - `CUDA_HOME=/usr/local/cuda CUDA_OXIDE_LLC=/usr/bin/llc-21 cargo oxide
       run lowp_roundtrip` in the B300 pod: passed; auto-detected `sm_103`
       and printed `SUCCESS: low-precision typed buffers and device
       conversions matched host references`.
     - Claude CLI non-interactive review: no blocking issues.

3. Inference-style packing API and integration checks
   - Status: complete
   - End-state: kernels have ergonomic helpers for moving low-precision
     vectors through router/indexer and accelerator-adjacent code without
     bespoke bit manipulation at each call site.
   - Implementation plan:
     - add small typed pack/unpack helpers for groups used by inference
       kernels, keeping byte order explicit and documented;
     - add compile checks that low-precision values work in `SharedArray`,
       slices, disjoint output slices, and kernel argument paths;
     - where CUDA exposes a direct storage-data-type mapping, add typed mapping
       helpers rather than open-coded enum constants in examples.
   - Validation:
     - Local `cargo fmt --check`: passed.
     - Local `cargo fmt --check --manifest-path
       crates/rustc-codegen-cuda/examples/lowp_roundtrip/Cargo.toml`: passed.
     - Local `cargo test -p cuda-lowp -- --nocapture`: passed; 8 unit tests.
     - Local `rustup run nightly cargo check -p cuda-device`: passed.
     - Local `git diff --check`: passed.
     - `CUDA_HOME=/usr/local/cuda cargo test -p cuda-lowp -- --nocapture` in
       the reusable B300 pod: passed; 8 unit tests.
     - `CUDA_HOME=/usr/local/cuda cargo check -p cuda-core` in the B300 pod:
       passed.
     - `CUDA_HOME=/usr/local/cuda cargo check -p cuda-device` in the B300 pod:
       passed.
     - `cargo fmt --check --manifest-path
       crates/rustc-codegen-cuda/examples/lowp_roundtrip/Cargo.toml` in the
       B300 pod: passed.
     - `CUDA_HOME=/usr/local/cuda CUDA_OXIDE_LLC=/usr/bin/llc-21 cargo oxide
       run lowp_roundtrip` in the B300 pod: passed; auto-detected `sm_103`
       and validated typed lowp `SharedArray`, lowp `DisjointSlice` outputs,
       and by-value fp8x4 kernel arguments.
     - Claude CLI non-interactive review: no blocking issues.

4. Docs and roadmap closure
   - Status: complete
   - End-state: README/book/support matrix describe the shipped low-precision
     type story and item 5 is marked complete on this board.
   - Implementation plan:
     - document the type/packing API in the root README, device/runtime crate
       docs, API quick reference, and supported-features matrix;
     - replace the `FP8 / MX Data Types` planned entry with the shipped scope;
     - keep the roadmap honest about storage/conversion support versus any
       future tensor-core or library-matmul expansion.
   - Validation:
     - Local example count check: `find crates/rustc-codegen-cuda/examples
       -maxdepth 1 -mindepth 1 -type d | wc -l` returned 62, matching the
       root README.
     - Local `cargo fmt --check`: passed.
     - Local `git diff --check`: passed.
     - `PATH=.venv/bin:$PATH make html` in `cuda-oxide-book`: passed.
