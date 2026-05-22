# topk_select

Block-cooperative top-k selection example.

One block scans one row of scores, using caller-provided shared memory scratch
for deterministic top-k selection. The host validates all output `(score,
index)` pairs against a CPU reference with deliberate score ties.

Run:

```bash
cargo oxide run topk_select
```
