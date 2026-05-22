# Warp MMA

Reference-checked `m16n8k16` warp MMA example.

This example computes one 16x8 output tile with K=32. The kernel stages two
16-wide f16 K tiles through shared memory, loads fragments with `ldmatrix`, and
accumulates them with `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`.

Run:

```bash
cargo oxide run warp_mma
```
