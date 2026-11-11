# mako-image

Image generation on Apple Silicon from the models' original files: Qwen-Image 2.1, Krea 2, MOSS-Image, MiniMax-H3-Image,
Anima and SDXL (single-file checkpoints such as Illustrious fine-tunes).
An OpenAI-compatible HTTP server (`mako_image_serve`) and a command-line tool (`mako_image`).

It runs beside [mako](https://github.com/jhanssen/mako) (LLM inference) or on its own. On a Mac that runs both and
has no room for both models at once, `--llm` makes the image server borrow mako's memory: before an image model loads,
mako finishes its requests in progress and unloads its LLM (its conversations stay in memory); when a chat request
comes, the image model is freed after the image in progress and mako loads its LLM again. On another machine, or with
room for both, leave `--llm` out.

## Requirements

- Apple Silicon Mac, macOS 26 or later
- Xcode with the Metal toolchain: `xcodebuild -downloadComponent MetalToolchain`
- CMake 4.3 or later, Ninja

## Build

```sh
git clone --recurse-submodules https://github.com/jhanssen/mako-image
cmake -S mako-image -B mako-image/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build mako-image/build
```

(`git submodule update --init` in a clone made without `--recurse-submodules`: the shared code is
[mako-common](https://github.com/jhanssen/mako-common), in `external/mako-common`.)

## Run

```sh
./build/mako_image_serve --image-models ~/models/image --port 8082
# sharing the Mac with mako_serve on port 8080:
./build/mako_image_serve --image-models ~/models/image --port 8082 --llm http://127.0.0.1:8080
```

Options: `--image-models DIR`, `--image-model NAME|DIR`, `--upscalers DIR` (ESRGAN upscalers, below; default
`--image-models`' `upscalers/`), `--host H`, `--port N` (default 8082), `--api-key
KEY[,KEY...]` / `--api-key-file FILE` (needed to listen beyond loopback), `--max-waiting N`, `--llm URL` and
`--llm-api-key KEY` (when mako_serve wants one). Each also from `MAKO_IMAGE_<OPTION>` (e.g. `MAKO_IMAGE_PORT`), and
from presets (`--preset NAME`, `MAKO_IMAGE_PRESET`).

`mako_image <model dir> <prompt> [-o out.png] ...` renders from the command line (`--help` lists its options; `--llm`
works the same; `--upscale FILE[:SCALE]` runs an ESRGAN upscaler on the result). `mako_upscale MODEL.safetensors IMAGE
[-o out.png] [--scale F]` upscales an image.

## API

`mako_image_serve --image-models DIR` serves `POST /v1/images/generations`, OpenAI's image API, with the image models
in DIR, one directory each (named by it); `--image-model NAME` is the one for requests that name none (default: the
first by name; or a model directory's path, alone without `--image-models`):

```sh
curl -s localhost:8082/v1/images/generations -H 'Content-Type: application/json' \
  -d '{"prompt": "a misty harbour town at dawn", "size": "1024x1024", "model": "qwen-image"}' \
  | jq -r '.data[0].b64_json' | base64 -d > out.png
```

- A model directory holds the model's files as published (Qwen-Image 2.1, Krea 2, MOSS-Image, MiniMax-H3-Image:
  `model_index.json` says which; Anima and SDXL: below), or a `mako.json` pointing at them, with its settings (every
  field optional, paths relative to the directory or from `~/`):

  ```json
  {"model": "~/models/qwen-image-2.1", "text_encoder": "~/models/qwen-image-2.1-te-heretic",
   "transformer": "...", "lora_dir": "../qwen-image/loras", "loras": {"lightning-4step": 1.0},
   "steps": 4, "terminal": 0, "guidance": 4, "sampler": "euler", "description": "what it is good at and its time",
   "prompting": "how to write its prompts"}
  ```

  `text_encoder`, `transformer` and `vae` take another copy of those weights (a refusal-ablated text encoder, a
  checkpoint with LoRAs merged, a repack's VAE); `loras` are added for requests that name none (by name in `lora_dir`, default the directory's
  `loras/`; a scale, or `{"scale": 1.0, "merge": false}` to choose how it goes in, below); `steps` (0: the model's),
  `terminal` (the schedule's last noise level, 0: none, as few-step LoRAs want), `guidance`, `sampler` and
  `skip_blocks` set its defaults; `adapter` (MOSS-Image) a data-space distillation adapter, `control` (MOSS-Image) a control adapter file (both
  below); `min_area` (MiniMax-H3-Image) the smallest canvas in pixels (below). The directory is
  read on each request: models and LoRAs added later count.
- `GET /v1/images/models` lists the models: `id`, `family` (which model it is), `description`, `prompting`,
  `default`, `loaded`, `steps`, `guidance`, `sampler` and `skip_blocks` where set, `image_to_image`, `control` and
  `references` when it takes a starting image, control maps or reference images, `fixed_schedule` and
  `min_generation_area` where the model has them, and the default `loras` with their scales (and `merge` where set).
- MOSS-Image also takes a layout as its prompt: compact JSON with a caption for the whole image and a box per
  character, object or text (`{"summary": {"caption": ...}, "items": [{"label": "character", "caption": ...,
  "bbox": [y0, x0, y1, x1]}, ...], "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"}`, boxes in pixels
  from the image's centre). The skill's `render.sh --layout` builds it from boxes in pixels from the top-left.
- Fields: `prompt` (required, at most 16384 bytes, as is `negative_prompt`; Qwen-Image reads the first 1024 tokens),
  `model` (a name from that list), `n` (1-4), `size` (`WIDTHxHEIGHT` in multiples of 32
  from 256, at most 2048x2048 pixels, MiniMax-H3-Image 6272x2688; `auto`: 1024x1024), `response_format` (`b64_json` only: images are not stored),
  `output_format` (`png` or `jpeg`; a PNG carries how it was made in text chunks: `parameters`, the prompt, negative
  prompt, steps, CFG scale, seed, size, model and LoRAs as image tools commonly read them, and `mako`, the same as
  JSON). `quality`, `style`, `background`, `moderation` and `user` are accepted and left
  alone: the model's steps (Qwen-Image 2.1 40, Krea 2 Turbo 8, MOSS-Image 50, MiniMax-H3-Image 6, Anima Turbo 10,
  Anima 30, SDXL 28), a random seed for each image (in the log) unless `seed` gives one.
- Extensions: `steps` (0 to 100; 0: the model's own), `negative_prompt` (what the image should not be) and
  `guidance_scale` (its strength; default the model directory's, else the model's: Qwen-Image 2.1 4, Krea 2 Turbo 2,
  MOSS-Image 3, Anima 4, SDXL 6), `sampler` (`euler`; `euler_a`, Euler ancestral, for SDXL only; `er_sde`, the
  three-stage extended reverse-time SDE solver, which adds noise each step; default the model directory's, else the
  model's: Euler, SDXL Euler a), `skip_blocks` (MOSS-Image: the blocks the negative prompt's pass leaves out, e.g. `[22,
  24]`; default `[22]`, as its reference), `control` (below), `image` (MOSS-Image: an image file in base64, or a `data:` URL, to
  start from: scaled to cover the size and centred; without `size`, its own size scaled down to 1024x1024 pixels' area
  when larger, in multiples of 32) with `strength` (above 0, at most 1; default 0.75: the share of the steps run on it,
  rounded to whole steps, below 1 at least the first left out; the image noised to the first one's level; more
  changes more; the PNG records the share run), `seed` (a whole number, as a string past 2^53, as the PNG records it; the
  images of one request take seed, seed + 1, ...; the same seed and settings give the same image) and `loras` (below). A negative prompt runs the transformer twice a step, so
  the image takes about twice as long; without one, nothing changes. Qwen-Image follows it well (scales 2 to 6 tried).
  Krea 2 Turbo was distilled to run without it: at 1.5-2 the negative prompt shows a little, at 3 the image turns flat
  and oversaturated, at 5.5 it breaks. MOSS-Image is always guided, as its reference runs it: against an empty negative
  prompt when the request gives none, that pass skipping block 22 (`skip_blocks`). Anima (not Turbo) is guided by
  default against its model card's negative prompt without the score tags the card advises against for it ("worst
  quality, low quality, artist name, blurry, jpeg artifacts, chromatic aberration"), SDXL against "bad quality, worst
  quality, worst detail, sketch, censor" (WAI-illustrious's recommended one). MiniMax-H3-Image is distilled to a
  six-step grid with one output head each and no guidance: `negative_prompt`, `guidance_scale` other than 1,
  `sampler` other than `euler` and `terminal` are refused; other `steps` interpolate its grid (as ComfyUI samples it:
  a step inside one grid interval takes that interval's head, a step across several the blend of theirs weighted by
  the sigma each covers), off the schedule it was distilled for, so judge the result by eye. Its extensions: `references`, an array of
  image files in base64 (or `data:` URLs; up to 16, 32 pixels a side to 16 megapixels each) the prompt names in that
  order as `<Picture 1>`, `<Picture 2>`, ... (an edit: say what to keep, what to change, then describe the whole
  target image), each taken at its own size (resized only when its short edge is outside 1024 to 2048 pixels), and
  `min_generation_area`: the model generates on a canvas of at least this many pixels (default 4194304, 2048x2048,
  as released, where it was trained; 4 to 20 megapixels is its working range, best around 16: the card's 16 MP sizes
  are 4096x4096, 4736x3552, 5024x3360, 5472x3072, 6272x2688 and their portrait swaps, each side sqrt(16777216 x
  ratio) rounded to a multiple of 32) at the aspect ratio asked for and
  resizes the image to `size` (Lanczos); `0` generates at `size` itself (a 512x512 draft in 13 s instead of 37).
  The PNG records the canvas (`Canvas`) and the number of references.
- One model is in memory at a time, loaded by the first request for it; a request for another model or other LoRAs
  than the loaded ones loads those first. `GET /v1/status` says which is loaded.
- Time: Qwen-Image 2.1 ~13 s for 512x512, ~63 s for 1024x1024 (40 steps); Krea 2 Turbo ~5 s and ~23 s (8 steps);
  MiniMax-H3-Image ~37 s for 1024x1024 (6 steps on its 2048x2048 canvas, 4.3 s a step, the first 14), ~13 s for
  512x512 generated at that size (`min_generation_area` 0);
  MOSS-Image ~32 s and ~145 s (50 steps, two passes each: 2.3 s a step cool, 3.4 s once warm); Anima Turbo 4.5 s for
  1024x1024 (10 steps), Anima Aesthetic 27 s (30 steps, two passes each), SDXL (WAI-illustrious v17) ~13 s for
  1024x1024 or 832x1216 (28 steps, two passes each); slower as the Mac heats up; the log has
  each image's progress.
- LoRAs: `GET /v1/images/loras?model=NAME` (default: the default model) lists what a request can name for a model: its
  default ones first, with their `default_scale`, then its LoRA directory's, each by its file name without
  `.safetensors`, with a `trigger` (words a prompt needs for the LoRA to act), a `description` and a recommended
  `scale_range` where known: from `NAME.json` beside the file (`{"trigger": "...", "description": "...", "scale_range":
  [0.8, 1.4]}`), else the trigger and description from the file's metadata (`trigger_word` or
  `modelspec.trigger_phrase`; `modelspec.description`). Few files carry these (of four Qwen-Image LoRAs tried, one has a
  trigger). The trigger is not added to prompts: where it goes differs. A request's `loras`, e.g. `[{"name":
  "next-scene", "scale": 0.8}]` (scale 1 when left out, -10 to 10; `"merge": true|false` below), replaces the model's
  default set for that request; `[]` is none. A LoRA is merged into the BF16 weights, or kept beside them (`merge:
  false`) and applied on every pass at about 15-30% more time a step for a rank-128 one; MOSS-Image keeps the ones on
  its blocks' projections unmerged by default, the other models merge (and refuse `merge: false`). Distillation LoRAs
  change the weights by less than BF16's rounding: merged, lodestones' 8-step DMD LoRAs and the MOSS Image Turbo LoRA
  come out 79-97% off their change in the matrices measured (a style LoRA 9-23%). A set without a few-step LoRA the
  model's `steps` were lowered for gives smeared images unless the request raises `steps` too. Formats: Diffusers, PEFT,
  ComfyUI and Kohya (`lora_unet_...`) names, whole bias differences (`diff_b`), adapters of fused matrices (MOSS-Image's
  `proj.qkv`). Tried: Qwen-Image's lightx2v Lightning (4 steps, 1024x1024 in ~7 s), Next-Scene and Natural-Exposure
  stacked with it; lodestones' 8-step DMD LoRAs for MOSS-Image SFT and Base (`steps` 8, `guidance` 1: 1024x1024 in
  ~11-13 s unmerged, ~10 s merged, instead of ~145 s; merged and unmerged images differ, which is better is untested);
  five Krea 2 and two more MOSS-Image LoRAs load (their images not checked); two Illustrious style LoRAs (Kohya's, both
  text encoders and the UNet) against the reference with one merged at 0.8. Guidance 1 runs no negative pass.
- Anima (circlestone-labs/Anima) as its repository lays it out: the directory holding `split_files/` (or
  `diffusion_models/`, `text_encoders/`, `vae/` directly) with `transformer` the checkpoint (e.g.
  `split_files/diffusion_models/anima-turbo-v1.1.safetensors`; a name with "turbo" is run as the distilled one), and
  the two tokenizers it does not ship, in `tokenizers/` beside it:
  `hf download Qwen/Qwen3-0.6B-Base tokenizer.json --local-dir DIR/tokenizers/qwen3` and
  `hf download google/flan-t5-xxl tokenizer.json --local-dir DIR/tokenizers/t5`. Prompts take ComfyUI's weights
  (`(chibi:2)`); its `embedding:` syntax is not supported. Its sampler is Euler; the model card's default, er_sde, is
  `sampler: "er_sde"` (not compared with the reference's images).
- MOSS-Image data-space adapters (`adapter` in `mako.json`, `--adapter`): a directory with `lora.safetensors` (kept
  unmerged), `time_embedding.safetensors` (the time path's weights in F32, run on the CPU) and
  `dataspace.safetensors` (a latent mixing matrix and the steps' timesteps): each step runs the transformer on
  z + Q x0 and takes x0 = z - its output, from one fixed noise z; no guidance. Tried: akhaliq's IDM 2-step adapter
  (`runs/idm/export/000750` of akhaliq/MOSS-Image-IDM-2-Steps, on SFT): 1024x1024 in 3.4 s, clean; not compared
  with its reference.
- SDXL as a single-file checkpoint (the original names: `model.diffusion_model.`, `conditioner.embedders.`,
  `first_stage_model.`), F16 or F32, the directory's one `.safetensors` (or `transformer` naming it), with SDXL's
  tokenizer beside it: `hf download stabilityai/stable-diffusion-xl-base-1.0 --include 'tokenizer/*' --local-dir DIR`.
  Tried with WAI-illustrious-SDXL v17 (noise prediction). Prompts take ComfyUI's weights and its 75-token chunks;
  `embedding:` (textual inversions) is not supported. V-prediction checkpoints are not supported. LoRAs: Kohya's
  names (`lora_unet_`, `lora_te1_`, `lora_te2_`) of linear layers; convolution LoRAs (LoCon, LyCORIS) are not.
- `tools/pi/scene-image` is a pi skill that turns the conversation into an image prompt and renders it through this
  endpoint (`ln -s "$PWD/tools/pi/scene-image" ~/.pi/agent/skills/`; then ask for an image of the scene, or
  `/skill:scene-image`). It lists the models and picks one by their descriptions, lists the chosen model's LoRAs and
  passes them when the user wants them. For MOSS-Image it can lay subjects out in boxes. For tag-based models it looks
  tags up in Danbooru's tag list and checks the prompt's (`scripts/tags.py`; the list as a1111-sd-webui-tagcomplete ships it, `name,category,posts,"aliases"`, at
  `~/models/tags/danbooru.csv` or `MAKO_TAGS_CSV`). Its script saves to `~/Pictures/mako` (`MAKO_IMAGE_DIR`) and opens
  the image; `MAKO_IMAGE_URL` (default `http://127.0.0.1:8082/v1`) and `MAKO_IMAGE_API_KEY` point it at the server.
  The scene-3d skill (a repository of its own) builds a control model's maps from a 3D blockout of the scene in
  Blender and renders through this skill.
- Control (MOSS-Image, a model directory with `"control"`, a colour/depth control adapter such as
  darask0/MOSS-Image-color-depth-control): `control` in a request, `{"color": BASE64, "depth": BASE64, "color_end": F,
  "depth_end": F}`: a colour region map (a flat colour per subject) and a depth map (near white), each steering the
  steps before its end (the share run, default 1). The maps go through the VAE encoder; the adapter widens the image
  input to [latents | colour | depth] and brings a LoRA (its blocks' part kept unmerged, its text refiner's merged).
  With the 8-step DMD LoRA an end of 0.25 kept poses and placement without the map's shapes showing in the bodies.
- Upscaling with ESRGAN models (the original's and Real-ESRGAN's RRDBNet; other architectures such as DAT, HAT,
  SPAN or SwinIR are not read) from `--upscalers DIR` (default: `--image-models`' `upscalers/`), `NAME.safetensors`
  each, converted from the .pth checkpoints (or a safetensors in the original names) by
  `tools/esrgan_to_safetensors.py MODEL.pth [OUT.safetensors]` (no torch needed; the original ESRGAN's layer names
  become Real-ESRGAN's), with an optional `NAME.json` beside it (`{"description": "..."}`, what it is for).
  `GET /v1/images/upscalers` lists them (`name`, `factor`, `blocks`, `description`). Tried, on a cartoon render and a
  Krea photograph at 4x: for drawn images 4x-AnimeSharp (clean edges) and 4x-UltraSharp (keeps texture),
  RealESRGAN_x4plus_anime_6B flattening fills; for photographs 4x_foolhardy_Remacri and 4x_NMKD-Siax_200k (skin
  texture and bokeh kept), 4x-UltraSharp close behind, while 4x_NMKD-Superscale amplifies grain everywhere,
  4xNomos8kSC streaks the bokeh, RealESRGAN_x4plus and 4x_RealisticRescaler smooth skin, and the anime models paint
  photographs flat. None changes the style: a drawn image stays drawn. `POST /v1/images/upscales` with `image` (base64 or a `data:` URL, at most 2048x2048 pixels'
  area), `model` (default: the first by name), `scale` (1 to the upscaler's factor, default the factor; less is
  resampled down from its output with Lanczos) and `output_format` answers like a generation, `size` the result's
  (no provenance chunks). A generation's `upscale` (a name, or `{"model": NAME, "scale": F}`) upscales its images
  before they are encoded; the PNG records it (`Hires upscale`, `Hires upscaler`; `mako`'s `upscale`). The result is
  at most 4096x4096 pixels' area. The weights run as F16 (0.8-2.2e-4 off PyTorch's fp32 output on a 48x32 crop);
  images over 1024 pixels a side run in 1024-pixel tiles with 32 of context. 1024x768 at 4x takes 1.1 s with the
  23-block models and 0.5 s with the 6-block one (ComfyUI's MPS path 3.1 s and 1.5 s; the convolutions read the image
  through shifted views instead of unfolding it, docs/architecture.md). Loaded upscalers stay in memory (33 MB
  each); their scratch buffers (about 11 GB for 1024x768 at 4x) are freed after each image. `mako_upscale MODEL.safetensors IMAGE [-o
  out.png] [--scale F] [--tile N] [--overlap N]` does the same from the command line, `mako_image --upscale
  FILE[:SCALE]` after a generation, the skill's `render.sh --upscale NAME[:SCALE]` through the server
  (`scripts/upscalers.sh` lists them).
- MiniMax-H3-Image (MiniMaxAI/MiniMax-H3-Image, text to image and editing with reference images; 33 B parameters,
  the H3 video model's transformer without its audio stream, its Qwen3-VL-32B text encoder unchanged): a directory
  with the release's small files and either the release's `transformer/` shards (66 GB BF16, with
  `pdd_fused_heads.safetensors`, the per-step output heads) or the ComfyUI repack's pruned single file
  (Comfy-Org/MiniMax-H3-Image, `minimax_h3_image_pruned_bf16.safetensors`, 40 GB: the time embedder and the blocks'
  2688-wide modulation projections folded into an 8-coordinate curve table, the heads stacked in its own output
  head), named by `transformer`; the VAE as the release's `vae/` (fp32, 2.3 GB; its `vae_diffusers/` copy under
  diffusers' tensor names is not read) or the repack's fp16 file (`vae`); the text encoder MiniMax-H3's (`hf download MiniMaxAI/MiniMax-H3 --include 'text_encoder/*'`, 67 GB,
  linked or named by `text_encoder`; its first 50 layers load, 49 GB). The repack's int8 and NVFP4 files are not
  read (ComfyUI's own formats). In memory: the text encoder and the pruned transformer together, 89 GB, so a 128 GB
  Mac runs it beside nothing else large (`--llm` borrows mako's). With the release's shards (66 GB) the two do not
  fit together with room there (the rule: 1.35 x their files against the machine's memory), so they take turns: the
  text encoder is freed when the transformer loads and loaded again for the next prompt, the transformer freed
  then, about 115 GB read from the SSD per request. The prompt goes in verbatim (no template; a long
  English description of the whole image works best, as the model's prompt expander writes them: every hard fact
  kept, required text placed and quoted, camera, lighting and each body's orientation stated, about 250 words for
  a scene; an edit as the expander's three parts, each kept source element defined as `<Subject N>` from its
  `<Picture N>`, then `The target image shows ...` with one preservation clause per subject, then only what
  changes; the first 4096 tokens; up to 16 references by the card). Its own schedule: six Euler steps on a sigma
  grid, each through its own output head, other counts interpolating the grid, no guidance; LoRAs merge into the
  transformer as it loads (Diffusers, PEFT and the checkpoint's split q / k / v names; H3 LoRAs trained in ComfyUI or
  Kohya tools name a fused `attn.qkv_proj` module, which is refused, and LoKr files are not read). Tried: the pDMD
  2-step distillation LoRA (pdmd2026's, trained on the video model, akhaliq's Diffusers-named repack for this one):
  with `steps` 2 a finished 1024x1024 photograph where the plain 2 steps come out soft. The video model's other LoRAs
  load but its weights sit 30-45% away from this model's.
- Not yet: `/v1/images/edits` (MiniMax-H3-Image's references go through `references` of a generation), condition
  images for the other models, and image to image for the other models (their VAEs' encoders).


## Tests

```sh
ctest --test-dir build -L cpu    # the server and the memory hand-over, no GPU or model
ctest --test-dir build -L image  # the models against their references (the variables below)
```

The `cpu` tests include `er_sde`: the ER-SDE sampler over a made-up model against the reference sampler's numbers.

| Variables (CMake cache or environment at the first configure) | Tests |
| --- | --- |
| `MAKO_QWEN_IMAGE`, `MAKO_QWEN_IMAGE_REF` | `image_*`, `safetensors` |
| `MAKO_KREA2`, `MAKO_KREA2_TRANSFORMER`, `MAKO_KREA2_REF` | `krea_*` |
| `MAKO_MOSS`, `MAKO_MOSS_REF` | `moss_*` |
| `MAKO_ANIMA`, `MAKO_ANIMA_TRANSFORMER`, `MAKO_ANIMA_REF` | `anima_*` |
| `MAKO_SDXL`, `MAKO_SDXL_REF`, `MAKO_SDXL_LORA` | `sdxl_*` |
| `MAKO_H3_IMAGE`, `MAKO_H3_IMAGE_REF`, `MAKO_H3_IMAGE_PRUNED` (the repack's transformer) | `h3_image_*` |
| `MAKO_UPSCALERS`, `MAKO_ESRGAN_REF` | `esrgan` |
| `MAKO_T5_TOKENIZER` (a T5 `tokenizer.json`) | `unigram` |

The image tests compare with reference tensors from the original pipeline (fp32 on the CPU), written by a Python
script kept beside the model, not in this repository; MiniMax-H3-Image's by `tools/dump_h3_image_ref.py` (the Diffusers
implementation: the text encoder in fp32 on the CPU, the release's transformer with its BF16 weights in fp32 on MPS,
the VAE in fp32), the `h3_image_pruned_*` tests running the repack's transformer against the same tensors: text
embeddings within 3e-5, the transformer's velocities within 0.4-2% a step (the pruned file 0.5-1%), the decoded
image within 6e-4 (a level or two off in 6% of the 8-bit samples), the encoder's latents within 3e-3; an edit's
presentation (two reference pictures) within 5e-3 and its velocities within 4% (2-3% with the reference's own
latents: the engine's F16 VAE encode of the references moves the output more than the transformer does). The edit
tests pass the pipeline's resized reference pictures in and check the engine's Lanczos resize at the pixel level on
its own (Qwen3-VL's vision tower moves 30% on the rounding differences between two Lanczos implementations over
a smooth synthetic picture, so no resize but a bit-identical one keeps the tower's output comparable). `esrgan` compares the upscalers with PyTorch's output
(spandrel's RRDBNet, fp32) for a 48x32 crop of a render: `MAKO_ESRGAN_REF/NAME/` holds `manifest.json` (`width`,
`height`, `factor`), `input.f32` and `output.f32` (RGBA channels-last in [-1, 1]) for `MAKO_UPSCALERS/NAME.safetensors`,
all at once and in 16-pixel tiles.

More: [architecture](docs/architecture.md), [performance](docs/performance.md).
