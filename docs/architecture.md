# Image models, as mako-image implements them

They run from their original files (BF16 safetensors and configs), not a GGUF: the published GGUFs hold only the
transformer, as F16 (BF16 values below ~6.1e-5 lose bits there) or quantized. They share `image::Engine`: the text
encoder (Qwen3-VL, or Qwen3.5's hybrid), the VAE decoder (Qwen-Image's or FLUX.2's), the flow-matching Euler schedule,
the denoising loop with guidance, and LoRA merging. `Engine::open` picks the model from `model_index.json`, or takes
Anima's single-file layout (`diffusion_models/`, `text_encoders/`, `vae/`). The text encoders' BF16 GEMMs also
multiply what rounding their activations to bfloat left out (a second GEMM): the first token's outlier channels grow
to thousands and cancel in the last layers, and with activations in bfloat alone its final state came out a few
percent off.

### Qwen-Image 2.1

- Transformer (13.25 GiB): 32 layers, hidden 4096 (32 heads of 128), MLP ratio 3, 64 latent channels in and out,
  patch size 1, rotary positions on three axes (16 / 56 / 56), `causal_condition`. A 1024x1024 image is 4096 latent
  positions; a step is ~57 TFLOP (2 x 7B x 4096), compute bound like a prefill.
- Text encoder (16.3 GiB): Qwen3-VL; text part 36 layers, hidden 4096, 8 KV heads, interleaved multi-axis rotary
  positions; a 27-layer vision tower for image prompts (not used yet).
- VAE (1.26 GiB): 4 channels (RGBA), 64-channel latent, 16x spatial compression (sides multiples of 32), with the
  temporal settings of a video VAE.
- Schedule: dynamic shift; 40 steps.

### Krea 2

- Transformer (12B): 28 single-stream blocks over [text, image], hidden 6144 (48 query heads, 12 key / value heads
  of 128), a sigmoid gate on the attention output, SwiGLU 16384, RMS norms with zero-centred scales, one shared
  timestep modulation plus a learned table per block, rotary axes 32 / 48 / 48 (theta 1000; the text at the origin).
- Text: Qwen3-VL-4B's hidden states 2, 5, ..., 35 per token, fused by two blocks across the twelve layers, a learned
  mix, and two blocks across the tokens (2560 wide, 20 heads). The reference pads the text to 512 rows and masks the
  padding; mako runs only the prompt's rows, which gives the same result.
- VAE: the first Qwen-Image's (16 channels, 8x), the transformer's tokens its 2x2 patches; causal 3D convolutions
  reduced to their last time tap.
- Turbo (`is_distilled`): 8 steps, a fixed shift (mu 1.15), no guidance.

### MOSS-Image

- Transformer (5B): 42 single-stream blocks over [image, text], hidden 3072 (24 heads of 128), no biases, RMS norms
  (plain scales) modulated by one timestep vector shared by every block (shift, scale, gate for attention and the
  FFN), an FFN of 8192 with a bounded gated activation (4 tanh(g / 4) sigmoid(g) x 25 tanh(h / 25)), rotary positions
  on four axes of 16 frequencies each (the image centred, after the text's length on the second axis; the text along
  that axis), half-split rotation.
- Text: Qwen3.5-9B's last hidden state (after its final norm) of the chat-templated prompt with thinking opened.
  The encoder alternates three gated delta rule layers (a causal depthwise convolution over q / k / v, l2-normed
  q and k, a per-head decay, a 128 x 128 state per value head; GLM's delta rule scan runs it with the decay on every
  channel) with one attention layer (heads of 256, a sigmoid output gate, the first 64 values rotated). Two refiner
  blocks (with biases) bring it to 3072 wide.
- VAE: FLUX.2's small decoder: 32 channels, 8x, the transformer's tokens its 2x2 patches normalized by a batch norm's
  statistics; group norms, a single-head mid attention.
- Schedule: shift 5 on a linear grid, snapped to the 1000 training levels; 50 steps. Guidance 3 always, against an
  empty negative prompt unless one is given, that pass skipping block 22, as the reference runs it.

### Anima

- Transformer (2B, Cosmos-Predict2's): 28 blocks, hidden 2048 (16 heads of 128), each self-attention over the image
  (3D rotary positions: of 64 pairs, 22 for time, 21 each for rows and columns, bases 10000 x 4^(42/40) for the
  image's axes), cross-attention to the text (512 rows or more, zero padded) and a GELU MLP of 8192, each after a
  layer norm modulated by the timestep: per block and part, a down and up projection of the normed embedding plus
  one shared low-rank term. Patches of 2x2 of the 16-channel latent (a zero padding-mask channel dropped).
- Text: Qwen3-0.6B's final hidden states of the prompt as it is; an adapter (6 blocks, 1024 wide, heads of 64) whose
  queries are the prompt's T5 tokens (SentencePiece unigram, `text::Unigram`) attends to them. ComfyUI's prompt
  weights (`(text:1.5)`, `(text)` 1.1, nested, `\(` literal) split the prompt into parts tokenized apart; a part's
  weight scales its rows of the adapter's output.
- VAE: the first Qwen-Image VAE in its original names (mapped onto the Diffusers layout's), its latent statistics
  built in.
- Schedule: shift 3 over 1000 levels, steps evenly spaced from the top; Euler. Turbo (the file name says so): 10
  steps, no guidance; the others 30 at guidance 4 against the model card's negative prompt without its score tags.

