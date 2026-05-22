# Roadmap execution board

## Active roadmap item

### 1. Large-Model Memory Residency Controls

- Status: in-progress
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
   - Status: open
   - End-state: an executable example demonstrates the residency controls on a
     CUDA path, and a B300 validation run proves the relevant runtime behavior.
   - Validation: reusable B300 pod in `hou2-prod1`, exact command/log capture.

4. Docs and roadmap closure
   - Status: open
   - End-state: README/book/support matrix reflect the shipped capability and
     item 1 is marked complete on this board.
   - Validation: doc/source consistency review plus reviewer gate before commit.

## Remaining roadmap items

### 2. Production Dense Linear Algebra Integration

- Status: open
- Plan: not started. Write a milestone plan before implementation begins.

### 3. Warp-Scoped Matrix Multiply Acceleration

- Status: open
- Plan: not started. Write a milestone plan before implementation begins.

### 4. Scalable Device-Side Selection and Sorting Primitives

- Status: open
- Plan: not started. Write a milestone plan before implementation begins.

### 5. First-Class Low-Precision Inference Data Types

- Status: open
- Plan: not started. Write a milestone plan before implementation begins.
