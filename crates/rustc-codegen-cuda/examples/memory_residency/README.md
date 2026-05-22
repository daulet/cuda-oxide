# Memory Residency

Demonstrates `cuda-core` residency handles on a real kernel path:

- managed input and output with advice, prefetch, and stream attachment;
- mapped host memory read by the device;
- registered caller-owned host memory read by the device.

Run with:

```bash
cargo oxide run memory_residency
```
