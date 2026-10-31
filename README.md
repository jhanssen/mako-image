# mako-image

Image generation on Apple Silicon from the models' original files: Qwen-Image 2.1, Krea 2, MOSS-Image and Anima.
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
./build/mako_image_serve --image-models ~/models/image --port 8081
# sharing the Mac with mako_serve on port 8080:
./build/mako_image_serve --image-models ~/models/image --port 8081 --llm http://127.0.0.1:8080
```

Options: `--image-models DIR`, `--image-model NAME|DIR`, `--host H`, `--port N` (default 8081), `--api-key
KEY[,KEY...]` / `--api-key-file FILE` (needed to listen beyond loopback), `--max-waiting N`, `--llm URL` and
`--llm-api-key KEY` (when mako_serve wants one). Each also from `MAKO_IMAGE_<OPTION>` (e.g. `MAKO_IMAGE_PORT`), and
from presets (`--preset NAME`, `MAKO_IMAGE_PRESET`).

`mako_image <model dir> <prompt> [-o out.png] ...` renders from the command line (`--help` lists its options; `--llm`
works the same).

## API

`mako_image_serve --image-models DIR` serves `POST /v1/images/generations`, OpenAI's image API, with the image models
in DIR, one directory each (named by it); `--image-model NAME` is the one for requests that name none (default: the
first by name; or a model directory's path, alone without `--image-models`):

```sh
curl -s localhost:8081/v1/images/generations -H 'Content-Type: application/json' \
  -d '{"prompt": "a misty harbour town at dawn", "size": "1024x1024", "model": "qwen-image"}' \
  | jq -r '.data[0].b64_json' | base64 -d > out.png
```

- A model directory holds the model's files as published (Qwen-Image 2.1, Krea 2, MOSS-Image: `model_index.json`
  says which; Anima: below), or a `mako.json` pointing at them, with its settings (every field optional, paths
  relative to the directory or from `~/`):

  ```json
  {"model": "~/models/qwen-image-2.1", "text_encoder": "~/models/qwen-image-2.1-te-heretic",
   "transformer": "...", "lora_dir": "../qwen-image/loras", "loras": {"lightning-4step": 1.0},
   "steps": 4, "terminal": 0, "guidance": 4, "description": "what it is good at and its time",
   "prompting": "how to write its prompts"}
  ```

  `text_encoder` and `transformer` take another copy of those weights (a refusal-ablated text encoder, a checkpoint
  with LoRAs merged); `loras` are merged for requests that name none (by name in `lora_dir`, default the directory's
  `loras/`); `steps` (0: the model's), `terminal` (the schedule's last noise level, 0: none, as few-step LoRAs want)
  and `guidance` set its defaults. The directory is read on each request: models and LoRAs added later count.
- `GET /v1/images/models` lists the models: `id`, `family` (which model it is), `description`, `prompting`,
  `default`, `loaded`, `steps` and `guidance` where set, and the default `loras` with their scales.
- MOSS-Image also takes a layout as its prompt: compact JSON with a caption for the whole image and a box per
  character, object or text (`{"summary": {"caption": ...}, "items": [{"label": "character", "caption": ...,
  "bbox": [y0, x0, y1, x1]}, ...], "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"}`, boxes in pixels
  from the image's centre). The skill's `render.sh --layout` builds it from boxes in pixels from the top-left.
- Fields: `prompt` (required, at most 16384 bytes, as is `negative_prompt`; Qwen-Image reads the first 1024 tokens),
  `model` (a name from that list), `n` (1-4), `size` (`WIDTHxHEIGHT` in multiples of 32
  from 256, at most 2048x2048 pixels; `auto`: 1024x1024), `response_format` (`b64_json` only: images are not stored),
  `output_format` (`png` or `jpeg`; a PNG carries how it was made in text chunks: `parameters`, the prompt, negative
  prompt, steps, CFG scale, seed, size, model and LoRAs as image tools commonly read them, and `mako`, the same as
  JSON). `quality`, `style`, `background`, `moderation` and `user` are accepted and left
  alone: the model's steps (Qwen-Image 2.1 40, Krea 2 Turbo 8, MOSS-Image 50, Anima Turbo 10, Anima 30), a random
  seed for each image (in the log).
- Extensions: `steps` (0 to 100; 0: the model's own), `negative_prompt` (what the image should not be) and
  `guidance_scale` (its strength; default the model directory's, else the model's: Qwen-Image 2.1 4, Krea 2 Turbo 2,
  MOSS-Image 3, Anima 4), and `loras` (below). A negative prompt runs the transformer twice a step, so the image takes
  about twice as long; without one, nothing changes. Qwen-Image follows it well (scales 2 to 6 tried). Krea 2 Turbo
  was distilled to run without it: at 1.5-2 the negative prompt shows a little, at 3 the image turns flat and
  oversaturated, at 5.5 it breaks. MOSS-Image is always guided, as its reference runs it: against an empty negative
  prompt when the request gives none, that pass skipping one block. Anima (not Turbo) is guided by default against
  its model card's negative prompt without the score tags the card advises against for it ("worst quality, low
  quality, artist name, blurry, jpeg artifacts, chromatic aberration").
- One model is in memory at a time, loaded by the first request for it; a request for another model or other LoRAs
  than the loaded ones loads those first. `GET /v1/status` says which is loaded.
- Time: Qwen-Image 2.1 ~13 s for 512x512, ~63 s for 1024x1024 (40 steps); Krea 2 Turbo ~5 s and ~23 s (8 steps);
  MOSS-Image ~32 s and ~145 s (50 steps, two passes each: 2.3 s a step cool, 3.4 s once warm); Anima Turbo 4.5 s for
  1024x1024 (10 steps), Anima Aesthetic 27 s (30 steps, two passes each); slower as the Mac heats up; the log has
  each image's progress.
- LoRAs: `GET /v1/images/loras?model=NAME` (default: the default model) lists what a request can name for a model:
  its default ones first, with their `default_scale`, then its LoRA directory's, each by its file name without
  `.safetensors`, with a `trigger` (words a prompt needs for the LoRA to act), a `description` and a recommended
  `scale_range` where known: from `NAME.json` beside the file (`{"trigger": "...", "description": "...",
  "scale_range": [0.8, 1.4]}`), else the trigger and description from the file's metadata (`trigger_word` or
  `modelspec.trigger_phrase`; `modelspec.description`). Few files carry these (of four Qwen-Image LoRAs tried, one has
  a trigger). The trigger is not added to prompts: where it goes differs. A request's `loras`, e.g.
  `[{"name": "next-scene", "scale": 0.8}]` (scale 1 when left out, -10 to 10), replaces the model's default set for
  that request; `[]` is none. A set without a few-step LoRA the model's `steps` were lowered for gives smeared images
  unless the request raises `steps` too. Formats: Diffusers, PEFT, ComfyUI and Kohya (`lora_unet_...`) names, whole
  bias differences (`diff_b`), adapters of fused matrices (MOSS-Image's `proj.qkv`). Tried: Qwen-Image's lightx2v
  Lightning (4 steps, 1024x1024 in ~7 s), Next-Scene and Natural-Exposure stacked with it; lodestones' 8-step DMD
  LoRAs for MOSS-Image SFT and Base (`steps` 8, `guidance` 1: 1024x1024 in ~14 s instead of ~145 s, clean); five
  Krea 2 and two more MOSS-Image LoRAs load (their images not checked). Guidance 1 runs no negative pass.
- Anima (circlestone-labs/Anima) as its repository lays it out: the directory holding `split_files/` (or
  `diffusion_models/`, `text_encoders/`, `vae/` directly) with `transformer` the checkpoint (e.g.
  `split_files/diffusion_models/anima-turbo-v1.1.safetensors`; a name with "turbo" is run as the distilled one), and
  the two tokenizers it does not ship, in `tokenizers/` beside it:
  `hf download Qwen/Qwen3-0.6B-Base tokenizer.json --local-dir DIR/tokenizers/qwen3` and
  `hf download google/flan-t5-xxl tokenizer.json --local-dir DIR/tokenizers/t5`. Prompts take ComfyUI's weights
  (`(chibi:2)`); its `embedding:` syntax is not supported. Its sampler is Euler (the model card's default, er_sde,
  is not implemented).
- `tools/pi/scene-image` is a pi skill that turns the conversation into an image prompt and renders it through this
  endpoint (`ln -s "$PWD/tools/pi/scene-image" ~/.pi/agent/skills/`; then ask for an image of the scene, or
  `/skill:scene-image`). It lists the models and picks one by their descriptions, lists the chosen model's LoRAs and
  passes them when the user wants them. Its script saves to `~/Pictures/mako` (`MAKO_IMAGE_DIR`) and opens the image;
  `MAKO_IMAGE_URL` (default `http://127.0.0.1:8081/v1`) and `MAKO_IMAGE_API_KEY` point it at the server.
- Not yet: condition images (edits, references; `/v1/images/edits`), image to image, and a per-request seed.


## Tests

```sh
ctest --test-dir build -L cpu    # the server and the memory hand-over, no GPU or model
ctest --test-dir build -L image  # the models against their references (the variables below)
```

| Variables (CMake cache or environment at the first configure) | Tests |
| --- | --- |
| `MAKO_QWEN_IMAGE`, `MAKO_QWEN_IMAGE_REF` | `image_*`, `safetensors` |
| `MAKO_KREA2`, `MAKO_KREA2_TRANSFORMER`, `MAKO_KREA2_REF` | `krea_*` |
| `MAKO_MOSS`, `MAKO_MOSS_REF` | `moss_*` |
| `MAKO_ANIMA`, `MAKO_ANIMA_TRANSFORMER`, `MAKO_ANIMA_REF` | `anima_*` |
| `MAKO_T5_TOKENIZER` (a T5 `tokenizer.json`) | `unigram` |

The image tests compare with reference tensors from the original pipeline (fp32 on the CPU), written by a Python
script kept beside the model, not in this repository.

More: [architecture](docs/architecture.md), [performance](docs/performance.md).
