# Image models, as mako-image implements them

They run from their original files (BF16 safetensors and configs), not a GGUF: the published GGUFs hold only the
transformer, as F16 (BF16 values below ~6.1e-5 lose bits there) or quantized. They share `image::Engine`: the text
encoder (Qwen3-VL, or Qwen3.5's hybrid), the VAE decoder (Qwen-Image's or FLUX.2's), the flow-matching Euler schedule,
the denoising loop with guidance and its samplers (Euler, Euler ancestral, ER-SDE in `image::ErSde`, a data-space
adapter's), and LoRA merging. `Engine::open` picks the model from `model_index.json`, or takes
Anima's single-file layout (`diffusion_models/`, `text_encoders/`, `vae/`) or an SDXL checkpoint (one file). The text
encoders' BF16 GEMMs also multiply what rounding their activations to bfloat left out (a second GEMM): the first token's
outlier channels grow to thousands and cancel in the last layers, and with activations in bfloat alone its final state
came out a few percent off.

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
  that axis), half-split rotation. Its weights from the model directory or a single-file checkpoint
  (`--transformer`, e.g. a fine-tune): q, k and v as one `proj.qkv` matrix or three, F16 matrices kept as stored (on
  the GEMM, as SDXL's).
- Text: Qwen3.5-9B's last hidden state (after its final norm) of the chat-templated prompt with thinking opened.
  The encoder alternates three gated delta rule layers (a causal depthwise convolution over q / k / v, l2-normed
  q and k, a per-head decay, a 128 x 128 state per value head; GLM's delta rule scan runs it with the decay on every
  channel) with one attention layer (heads of 256, a sigmoid output gate, the first 64 values rotated). Two refiner
  blocks (with biases) bring it to 3072 wide.
- VAE: FLUX.2's small decoder: 32 channels, 8x, the transformer's tokens its 2x2 patches normalized by a batch norm's
  statistics; group norms, a single-head mid attention. Its encoder (loaded with the first image to start from) is
  the same parts down: stride-2 convolutions padded only below and right, the posterior's mean packed and normalized
  as the decoder's input. Image to image runs the schedule's last `strength` share of steps from the image noised to
  the first of them, (1 - sigma) image + sigma noise.
- Control adapter (`Options::control`): its `video_embedder` (3072 x 384) replaces the image input, each token's row
  [latents | colour map latents | depth map latents] (the maps through the VAE encoder, zeros when not given or past
  their end), and its LoRA adds to the blocks (unmerged) and the text refiner's blocks (merged: its changes are 3-8%
  of the weights, BF16 rounding loses 2-5% of them). Both passes of guidance see the maps.
- Schedule: shift 5 on a linear grid, snapped to the 1000 training levels; 50 steps. Guidance 3 always, against an
  empty negative prompt unless one is given, that pass skipping block 22, as the reference runs it (other blocks:
  `skip_blocks`).
- LoRAs on the blocks' projections stay unmerged by default: each file's factors side by side per projection (down
  [R][k] and up [n][R] in BF16, R its ranks padded to 64), two GEMMs and an add after the projection. Distillation
  LoRAs change the weights by 0.02-0.06% of their norm, below BF16's rounding (about 0.4%): merged, they come out
  79-97% off their change.
- Data-space adapters (`Options::adapter`): the LoRA unmerged, the time embedding's MLP and the two modulation biases
  replaced by the adapter's F32 ones (the MLP run on the CPU, a 256 -> 3072 -> 3072 product a step), and the
  sampler: from one noise z and x0 = 0, each step runs the transformer on z + Q x0 (Q mixes the 128 latent channels)
  at the next step code's timestep and takes x0 = z - its output; no guidance.

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

### SDXL

- One checkpoint file (F16) in the original names; Engine's text encoder is not used, its VAE decoder is (the
  AutoencoderKL layout, mapped from the original names: 4 channels, 8x, a token its 2x2 patch, latents scaled by
  0.13025).
- Text: CLIP ViT-L (12 layers, 768, quick GELU) and OpenCLIP bigG (32 layers, 1280, GELU), causal, heads of 64, each
  chunk of 77 tokens on its own; the second to last layer's states of both side by side (2048 per token, chunks one
  after another), and bigG's final state at the first end token, layer normed and projected (1280). Prompt weights
  as ComfyUI applies them: a weighted token's states move away from the empty prompt's by its weight; a weighted
  part shorter than 8 tokens is not split across chunks. Tokenizer: CLIP's BPE (NFC, whitespace runs as one space,
  lowercase, word ends marked), `image::ClipTokenizer`.
- UNet (2.6B): input blocks at 320, 640, 1280 channels (two residual blocks each, the last two levels with
  transformers of 2 and 10 blocks, stride-2 convolutions between), a middle block (10), output blocks with the input
  blocks' outputs concatenated and nearest 2x upsampling; residual blocks add the timestep embedding's projection
  (all blocks' in one GEMM a step) after their first convolution; transformers: group norm, linear in, self- and
  cross-attention (the text's keys and values cached per block at begin()), GEGLU FFN, linear out. Added
  conditioning: the pooled state and the size (height, width, crop 0, 0, target height, width) as 256 sinusoid values
  each. F16 weights on the GEMM, which also runs the embeddings' single rows (there is no F16 GEMV).
- Schedule: 1000 levels from betas linear in their square root (0.00085 to 0.012); the "normal" schedule (levels
  evenly spaced in timestep, interpolated in log sigma), the UNet's timestep the nearest level; inputs scaled by
  1 / sqrt(sigma^2 + 1), the start noise by sqrt(1 + sigma_max^2). Euler, or Euler ancestral (the default): down to
  sqrt(next^2 - up^2), then fresh noise times up = next sqrt(1 - next^2 / sigma^2).

### MiniMax-H3-Image

- Transformer (33B): MiniMax-H3's (`models::H3` in mako-common, shared with mako-video), 50 blocks of full
  self-attention over one packed sequence [text | reference images | image] (hidden 5376, 56 heads of 128, SwiGLU
  14336), each row modulated by the AdaLN parameters of its (timestep, modality); a two-block token refiner on the
  projected text; three-axis rotary positions: text rows at (i, 0, 0), each reference image one unit on the time
  axis after the text's length, the generated image the unit after the last reference, the spatial axes the latent
  grid centred on the unit square times 32. Without the video model's audio stream, and with one F32 output head
  per step of its distilled schedule (the release's `pdd_fused_heads.safetensors`, or stacked in the ComfyUI
  repack's `final_layer.video_out`: the first whole, the others as offsets from it). The repack is pruned: its time
  embedder and the blocks' 2688-wide modulation projections are folded into an 8-coordinate curve table over t in
  [0, 1] and [96768][8] F16 projections (mako-video checked the mapping against the full video model at 2e-4). A
  token is a 2x2 patch of the 24-channel latent (96 values); the engine's latents are one latent a token, packed and
  unpacked around each step.
- Text: Qwen3-VL-32B's 50th layer output (before any norm; the first 50 of its 64 layers load, 49 GB), the prompt's
  tokens verbatim (no template, no special tokens around it, the first 4096). An edit's presentation puts a
  `Picture i: ` label and a vision block (start, the merged tokens' pads, end) per reference before the prompt; the
  block's rows carry the image modality tag for the transformer's modulation. The vision tower (`models::QwenVision`)
  sees each reference at a 1024 short edge.
- References: each at its own size for the VAE (resized, keeping its ratio, only when its short edge is outside
  1024 to 2048 pixels; cropped at the top left to whole latents, the latents to whole patches), its posterior
  sampled and the sample rounded to fp16 as the release does, then mixed with noise to t = 0.999 and held there for
  every step ahead of the generated rows. The posterior's and the mixing noise come from the request's seed.
- VAE (`image::H3ImageVae`, its own runtime; the release's `vae/` file or the repack's fp16 one, the model's own
  tensor names, so the release's `vae_diffusers/` copy is not read): the video VAE's causal 3D CNN encoder over one
  frame (its last time taps: group norms, SiLU, reflect-padded 3x3 convolutions in six levels of two residual blocks, a stride-2
  downsample after the first four, 16x down to 48 moments, a 1x1 quant convolution) and a 2D ViT decoder: a 1x1
  post-quant convolution, a token per latent pixel (24 -> 1024) plus four learned register tokens and a zero token,
  24 pre-norm blocks (RMS norms, 16 heads of 64 with unscaled RMS norms on q and k and two-axis rotary positions
  over half of each head at base 100, the coordinates centred on [-1, 1), residuals scaled per channel, a gated
  SiLU FFN of 4096), a layer norm and a projection to each token's 16x16 pixel patch. Pixels are ImageNet-normalized
  RGB over [0, 1]. Both run in 256-pixel tiles with overlaps of at least 64 blended linearly (the slack spread over
  the overlaps in whole latents), as the model shipped, so a larger tile changes the image. Latents are normalized
  per channel by the release's mean and deviation. Weights F16 (the repack's, or the release's F32 rounded).
- Schedule: the release's grid 1, 0.9675, 0.8875, 0.75, 0.5328, 0.2320, 0, each sigma rounded through its float
  timestep 1 - sigma as the pipeline does; Euler in the velocity form (the transformer's velocity points at the
  data, the engine's loop steps along the noise, so the step negates it); no guidance, no other step count.
- Memory: the text encoder (49 GB) and the transformer (40 GB pruned, 66 GB as released) each in their own runtime;
  the transformer loads on first use. Where the two do not fit together with room (1.35 x their files against the
  machine's memory) they take turns: the text encoder (and the vision tower) is freed when the transformer loads
  and loaded again for the next prompt, the transformer freed then.
- Canvas: an image smaller than `min_area` pixels (2048x2048 by default, as released) is generated at the same
  aspect ratio on a canvas of that area (sides rounded to multiples of 32, Python's rounding) and resized to the size
  asked for with Lanczos after the pipeline's 8-bit truncation; the PNG records the canvas.

### ESRGAN upscalers

`image::Esrgan` runs RRDBNet (the original ESRGAN's and Real-ESRGAN's, from a safetensors in Real-ESRGAN's names that
`tools/esrgan_to_safetensors.py` makes of a checkpoint) over channels-last half images with a zero ring (a padded
row above and below, a pixel left and right) and zero slack rows before and after. Its 3x3 convolutions do not unfold
the image: `conv3x3_dense` (image.metal) runs the dense GEMM's cooperative matmul over nine views of the image
shifted by whole pixels, one per tap, accumulating into one tile, with the weights laid out [9][cout][cin] (cin
rounded up to 64 with zero weights); the ring's own outputs are garbage that the bias kernel zeroes, so the next
convolution's taps read zeros past the edges. A dense block's five convolutions share one wide buffer
[pixels][nf + 4 gc]: the block's input in the first columns, each output written after it by the bias + LeakyReLU
kernel (float in, half out), so each reads its concatenated input in place; a convolution of 96 or 160 channels reads
up to 128 or 192, the extra columns holding the earlier run's finite values against zero weights. The residuals are
scale_add with a vector of 0.2; the upsampling stage doubles each level by nearest into the next level's padded
image. Images past 1024 pixels a side run in tiles with 32 pixels of context. Against PyTorch's fp32 output the F16
path is 1-2e-4 off on a 48x32 crop. 1024x768 at 4x: 1.0 s on the GPU with 23 blocks, 0.3 s with 6 (ComfyUI on MPS 3.1
and 1.5 s); unfolding the image (im2col to half rows, the VAEs' path in `image::ConvRuntime`) took 5.9 s, its copies
about 600 GB of traffic, and 32-wide output tiles for the 32-channel convolutions beat 64-wide ones by 15 %.
