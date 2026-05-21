# cuda-oxide Principles

## Core Principles

### Use the best tool for each stage, but own the full pipeline

cuda-oxide should make pragmatic tool choices without surrendering architectural
clarity. The compiler relies on:

- `rustc` and Stable MIR for frontend correctness,
- Rust-native IR infrastructure for project-owned transformation stages,
- LLVM NVPTX for backend code generation where the ecosystem already has the
  strongest implementation.

The result should remain understandable, inspectable, and debuggable as one
coherent pipeline rather than a chain of opaque handoffs.

### Rust is the programming model, not a veneer over another one

cuda-oxide exists to let developers write GPU programs natively in Rust:

- no DSL in place of Rust,
- no foreign-language binding layer as the primary authoring model,
- one source language across host and device code,
- one build flow that preserves Rust's type system, ownership model, and
  generics as far into GPU programming as practical.

### Safety is a first-class goal

The project should push GPU programming toward stronger, explicit safety
boundaries while acknowledging that GPUs introduce their own execution and
memory-model subtleties. cuda-oxide should prefer APIs and abstractions that make
correctness easier to express, misuse harder to hide, and low-level escape
hatches explicit.

### Prefer project-fit abstractions over one-for-one CUDA imitation

cuda-oxide should expose the capabilities advanced GPU programs need, but it does
not need to mirror every CUDA API, runtime concept, or library surface exactly.
When a different Rust-shaped abstraction better fits the project, that is the
right design direction.

### Keep the system transparent to contributors

The compiler and runtime should stay legible enough that contributors can trace
how kernels move through the system, inspect intermediate representations, and
reason about behavior without treating the implementation as a black box. Rust
tooling, clear stages, and debuggable transformation boundaries are part of the
product quality.

## Project Values

- **Pragmatism:** reuse mature infrastructure where it is clearly the best fit,
  and invest project effort where cuda-oxide needs differentiated ownership.
- **Rust-native ergonomics:** preserve the feel of real Rust instead of making
  GPU programming read like a foreign embedded language.
- **Explicit correctness:** prioritize type safety, ownership-aware APIs, and
  clearly documented unsafe or hardware-specific edges.
- **Capability over mimicry:** support the workloads that matter without making
  surface compatibility with existing CUDA interfaces the primary goal.
- **Honesty about maturity:** cuda-oxide is experimental and should communicate
  that clearly while improving through real user feedback.
