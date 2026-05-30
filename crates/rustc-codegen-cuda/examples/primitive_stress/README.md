# primitive_stress

Small stress test for primitive scalar support in cuda-oxide.

It covers cases that are easy for a MIR importer or lowering pass to miss:

- `char` constants and casts.
- `u128` / `i128` constants, arithmetic, shifts, and ABI passing.
- `usize` / `isize` arithmetic.
- Rust integer bit methods (`rotate_left`, `rotate_right`, `count_ones`,
  `leading_zeros`, `trailing_zeros`, `swap_bytes`, `reverse_bits`).
- Rust integer saturating arithmetic (`saturating_add`, `saturating_sub`).
- Rust float math methods (`abs`, `copysign`, `floor`, `ceil`, `round`,
  `trunc`, `mul_add`, `powi`, `powf`, `sqrt`, `exp`, `exp2`, `ln`, `log2`,
  `log10`, `sin`, `cos`) plus the `core::f32::math` / `core::f64::math`
  free-function forms.

Run it with:

```bash
cargo oxide run primitive_stress
```

## How code reaches the GPU

The integer and bit kernels lower to plain LLVM intrinsics that `llc`
handles fine — the standard `.ll → .ptx → cuModuleLoad` path works.

The float-math kernel lowers to `__nv_*` libdevice calls (`__nv_sinf`,
`__nv_powf`, etc.). `llc` emits PTX with those calls unresolved, so
cuda-oxide:

1. Auto-detects the `__nv_*` calls in the lowered LLVM module.
2. Emits `primitive_stress.ptx` with the unresolved libdevice calls.
3. The example calls `cuda_host::ltoir::load_kernel_module(&ctx, "primitive_stress")`,
   which transparently:
     - `dlopen`s `libnvvm.so` and `libnvJitLink.so` from the CUDA Toolkit
       (via the [`libnvvm-sys`](../../../libnvvm-sys) and
       [`nvjitlink-sys`](../../../nvjitlink-sys) crates).
     - Compiles `libdevice.10.bc` to LTOIR via libNVVM.
     - Links `primitive_stress.ptx` and the libdevice LTOIR to an
       architecture-qualified cubin via nvJitLink.
     - Loads the cubin via `CudaContext::load_module_from_file`.

There are no external C tools, no symlinked `tools/` directory, and no
build-pipeline boilerplate to maintain per example. The same helper works
for any standalone project that depends on `cuda-host` and has the CUDA
Toolkit installed.

`CUDA_OXIDE_TARGET` selects the portable PTX generation target.
`cuda_host::ltoir::load_kernel_module` links the resulting cubin for the
executing CUDA context by default; `CUDA_OXIDE_LINK_TARGET` overrides that
link target for controlled builds.
`CUDA_OXIDE_LIBDEVICE`, `LIBNVVM_PATH`, and `LIBNVJITLINK_PATH` override
the corresponding discovery searches; without them the helper probes
`CUDA_HOME`, `CUDA_PATH`, `/usr/local/cuda`, and `/opt/cuda`.
