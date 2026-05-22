# Warp MMA Smoke

Compile-time smoke test for the warp-scoped `mma.sync` path.

This example exercises:

- `ldmatrix.sync.aligned.m8n8.x4.shared.b16` for A fragments;
- `ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16` for B fragments;
- `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`.

Run the compile-only path with:

```bash
cargo oxide build warp_mma_smoke --arch sm_80
```
