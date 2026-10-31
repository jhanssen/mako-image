# Performance

## Image models (2026-10-29)

Qwen-Image 2.1, 1024x1024 and 40 steps, each after a 5-minute cool-down: 63.0 s (step 2 1.28 s, step 40 1.77 s); the
reference pipeline on MPS in BF16 65.1 s (1.36 / 1.75 s). Both slow down with heat during a run while macOS reports
the nominal thermal state (pauses between steps restore the speed). From 79.9 s by:

- attention with each simdgroup on its own rows: 12.0 -> 7.1 ms a layer;
- the GEMMs run on a multiple of 256 rows (the first step's text + image rows took 64-row tiles, twice as slow);
- dense BF16 / F16 weights read by matmul2d straight from device memory: 55 -> 63 TFLOP/s (BF16 with F32
  accumulation is as fast as F16 on the M5);
- q / k / v and gate / up one GEMM each, with the heads' norms fused;
- the VAE's convolution weights in F16 on that GEMM.

The text's keys and values are cached after the first step. Krea 2 Turbo: 1024x1024 in 8 steps, 23 s. Qwen-Image
2.1 with a 4-step distilled LoRA (lightx2v's Lightning): 6.9 s.

