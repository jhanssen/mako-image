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

## MiniMax-H3-Image (2026-11-10)

The ComfyUI repack's pruned BF16 transformer (40 GB) and fp16 VAE, MiniMax-H3's text encoder (49 GB resident), on
the 128 GB M5 Max, `mako_image` from a cold start (the weights read from the SSD in 10 s):

| Request | Canvas | Rows a step | Step | Total |
| --- | --- | ---: | ---: | ---: |
| 512x512, `--min-area 0` | 512x512 | 256 | 0.37 s | 13 s |
| 1024x1024 (the default canvas, resized back) | 2048x2048 | 4096 | 4.3 s (the first 14) | 37 s |
| 1024x1024 edit, one 1024x1024 reference, `--min-area 0` (the server, the transformer loaded on the request) | 1024x1024 | 1024 + 1024 | 4.6 s (the first 9.5) | 48 s |

The first step pays for the text encoder's pass and the GPU's warm-up (an edit's also for the vision tower over the
reference, its VAE encode and the longer text sequence: 1100 rows for one 1024x1024 reference). A step at 4096 rows
is 50 blocks of BF16 GEMMs over 20 B parameters (164 TFLOP) at about 40 TFLOP/s, the same rate as the video model's
steps; the sequence is short enough that attention is a small part. The VAE decodes 2048x2048 (64 tiles of 256
pixels, 261 tokens each) in about a second; the Lanczos resize to 1024x1024 is on the CPU. Larger canvases scale
with the rows: 4096x4096 is 16384 rows, about four times a step. A reference's rows add to every step like
generated rows (an edit at 1024 + 1024 rows costs what a 2048-row generation does). Not measured yet: LoRAs.

The release's shards (66 GB BF16) with the same text encoder do not fit this machine together with room, so the
engine takes turns (the text encoder freed when the transformer loads, loaded again for the next prompt):

| Request | Rows a step | Step | Total | Of which reloads |
| --- | ---: | ---: | ---: | ---: |
| `mako_image` 1024x1024 (2048x2048 canvas), cold, thermal state fair | 4096 | 5.6 s (the first 14) | 44 s | the transformer 6 s |
| the server, 1024x1024 `--min-area 0`, first request after the load | 1024 | 1.3 s | 17 s | the transformer 6 s |
| the server, the next request | 1024 | 1.3 s | 21 s | the text encoder 4.5 s, the transformer 6 s |

The reloads come from the page cache here (the file just read); from the SSD they are the 66 and 52 GB at its rate.
The 5.6 s step against the pruned file's 4.3 s is not an A/B (a different thermal state, and the per-block
modulation projections run every step in the release's model).

