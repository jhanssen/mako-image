#!/usr/bin/env python3
# Reference tensors for mako-image's Qwen-Image 2.1 edit path (one reference picture), from the Diffusers
# implementation in fp32: the text encoder with the picture (CPU), the VAE encoder (MPS), then the transformer (MPS)
# over the first steps of a small target canvas.
#   python tools/dump_qwen_image_edit_ref.py <model dir> <out dir> [--size 256] [--steps 2]
# Writes manifest.json and raw float32 files (row-major, shapes in the manifest): the reference picture as the
# pipeline takes it (RGBA 8-bit, ed_ref0.u8), the text embeddings after the system turn with the pad rows marked
# (ed_text_embeds, ed_pad_mask), the picture's latents [h][w][64] normalized, and per step the target latents, the
# transformer's joint input and block outputs and its prediction for the target rows.
import argparse
import gc
import json
import os
import time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("model_dir")
ap.add_argument("out")
ap.add_argument("--size", type=int, default=256)
ap.add_argument("--steps", type=int, default=2)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
manifest = {"tensors": {}}
PROMPT = "Keep the scene of <image1> as it is, but turn the yellow disc into a bright orange sun with soft rays."
SYS = "Comprehend and analyze the provided prompt."
PROBE_BLOCKS = (0, 1, 15, 31)


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def save(name, t):
    t = t.detach().float().contiguous().cpu()
    t.numpy().tofile(os.path.join(args.out, name + ".f32"))
    manifest["tensors"][name] = list(t.shape)


def write_manifest():
    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"), indent=1)


def synth(w, h, seed):
    """A smooth test picture with an alpha channel: colour gradients, a disc and a bar."""
    from PIL import Image
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    rng = np.random.default_rng(seed)
    cx, cy, r = rng.uniform(0.3, 0.7) * w, rng.uniform(0.3, 0.7) * h, 0.2 * min(w, h)
    img = np.stack([xx / w, yy / h, 0.5 + 0.5 * np.sin(6.28 * (xx / w + yy / h))], axis=-1)
    disc = (xx - cx) ** 2 + (yy - cy) ** 2 < r * r
    img[disc] = (0.95, 0.85, 0.1)
    bar = (yy > 0.8 * h) & (xx > 0.1 * w) & (xx < 0.6 * w)
    img[bar] = (0.1, 0.1, 0.4)
    rgba = np.concatenate([img, np.ones((h, w, 1))], axis=-1)
    return Image.fromarray((rgba * 255).round().astype(np.uint8), "RGBA")


def calculate_dimensions(target_area, ratio):
    import math
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    return round(width / 32) * 32, round(height / 32) * 32


# ---------------------------------------------------------------------------------------------- text with the picture
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor  # noqa: E402
from diffusers.image_processor import VaeImageProcessor  # noqa: E402

ref = synth(400, 300, 11)
rw, rh = calculate_dimensions(1024 * 1024, ref.size[0] / ref.size[1])
proc = VaeImageProcessor(vae_scale_factor=16)
ref_sized = proc.resize(ref, height=rh, width=rw)  # RGBA, as the pipeline resizes the picture for both uses
np.array(ref_sized).tofile(os.path.join(args.out, "ed_ref0.u8"))
manifest["ed_ref0_size"] = [rw, rh]
manifest["ed_ref0_source_size"] = list(ref.size)
np.array(ref).tofile(os.path.join(args.out, "ed_ref0_source.u8"))
from PIL import Image  # noqa: E402
white = Image.new("RGB", ref_sized.size, (255, 255, 255))
white.paste(ref_sized, mask=ref_sized.getchannel("A"))
vision_image = white

processor = Qwen3VLProcessor.from_pretrained(os.path.join(args.model_dir, "processor"))
template = ("<|im_start|>system\n" + SYS + "<|im_end|>\n<|im_start|>user\n"
            "<image1><|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n")
text = template.format(PROMPT)
inputs = processor(text=[text], images=[vision_image], padding=True, padding_side="left", return_tensors="pt")
sys_tokens = processor.tokenizer("<|im_start|>system\n" + SYS + "<|im_end|>\n", return_tensors="pt")["input_ids"]
drop = int(sys_tokens.shape[1])
ids = inputs["input_ids"][0]
img_token = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
manifest.update({"prompt": PROMPT, "ids": ids.tolist(), "ed_prompt": PROMPT, "ed_drop_idx": drop, "ed_ids": ids.tolist(),
                 "ed_image_grid_thw": inputs["image_grid_thw"].tolist()})
log(f"text: {len(ids)} tokens, {int((ids == img_token).sum())} pads, grid {inputs['image_grid_thw'].tolist()}")

log("loading the text encoder (fp32, CPU)")
te = Qwen3VLForConditionalGeneration.from_pretrained(os.path.join(args.model_dir, "text_encoder"), dtype=torch.float32,
                                                     device_map="cpu")
te.eval()
grabbed = {}
norm = te.model.language_model.norm
handle = norm.register_forward_hook(lambda m, a, o: grabbed.__setitem__("pre_norm", a[0]))
vis = te.model.visual
vis.merger.register_forward_hook(lambda m, a, o: grabbed.__setitem__("merger", o))
for i, mg in enumerate(vis.deepstack_merger_list):
    mg.register_forward_hook(lambda m, a, o, i=i: grabbed.__setitem__(f"deepstack{i}", o))
mm_types = torch.tensor(processor.create_mm_token_type_ids([ids.tolist()]), dtype=torch.long)
with torch.no_grad():
    te.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], mm_token_type_ids=mm_types,
             pixel_values=inputs["pixel_values"].float(), image_grid_thw=inputs["image_grid_thw"], use_cache=False)
handle.remove()
embeds = grabbed["pre_norm"][0][drop:]
save("ed_vis_merger", grabbed["merger"])
for i in range(len(vis.deepstack_merger_list)):
    save(f"ed_vis_deepstack{i}", grabbed[f"deepstack{i}"])
log("vision features", tuple(grabbed["merger"].shape))
pad_mask = (ids[drop:] == img_token)
save("ed_text_embeds", embeds)
save("ed_pad_mask", pad_mask.float())
manifest["ed_text_rows"] = int(embeds.shape[0])
manifest["ed_slots"] = [int(pad_mask[: int(torch.nonzero(pad_mask)[0])].logical_not().sum())]
log("ed_text_embeds", tuple(embeds.shape), "slot", manifest["ed_slots"])
del te, grabbed
gc.collect()
write_manifest()

# ---------------------------------------------------------------------------------------------------------------- VAE
from diffusers import AutoencoderKLQwenImage21  # noqa: E402

dev = torch.device("mps")
vae = AutoencoderKLQwenImage21.from_pretrained(os.path.join(args.model_dir, "vae"), torch_dtype=torch.float32).to(dev)
lmean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
lstd = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
manifest["latents_mean"], manifest["latents_std"] = list(vae.config.latents_mean), list(vae.config.latents_std)
pixels = proc.preprocess(ref_sized, height=rh, width=rw).unsqueeze(2)  # [1][4][1][h][w] in [-1, 1]
with torch.no_grad():
    z = vae.encode(pixels.to(dev), return_dict=False)[0].mode().float().cpu()
zn = (z - lmean) / lstd
lh, lw = zn.shape[-2:]
save("ed_ref0_latents", zn[0, :, 0].permute(1, 2, 0))  # [h][w][64]
manifest["ed_ref0_grid"] = [int(lh), int(lw)]
log("reference latents", tuple(zn.shape))
del vae
gc.collect()
torch.mps.empty_cache()
write_manifest()

# -------------------------------------------------------------------------------------------------------- transformer
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImage21Transformer2DModel  # noqa: E402
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift, retrieve_timesteps  # noqa: E402

h = w = args.size // 16
N = h * w
sched = FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(args.model_dir, "scheduler"))
sigmas = np.linspace(1.0, 1 / args.steps, args.steps)
mu = calculate_shift(N, sched.config.get("base_image_seq_len", 256), sched.config.get("max_image_seq_len", 4096),
                     sched.config.get("base_shift", 0.5), sched.config.get("max_shift", 1.15))
timesteps, _ = retrieve_timesteps(sched, args.steps, "cpu", sigmas=sigmas, mu=mu)
manifest.update({"size": args.size, "latent_height": h, "latent_width": w, "ed_steps": args.steps, "seed": args.seed,
                 "ed_sigmas": sched.sigmas.tolist(), "ed_timesteps": timesteps.tolist(), "ed_mu": float(mu)})
g = torch.Generator().manual_seed(args.seed)
latents = torch.randn((1, N, 64), generator=g, dtype=torch.float32)  # packed target rows
ref_rows = zn[0, :, 0].reshape(64, lh * lw).transpose(0, 1)[None]  # [1][lh * lw][64]
img_shapes = [[(1, int(lh), int(lw)), (1, h, w)]]
img_mask = torch.cat([pad_mask[None], torch.ones(1, N // 4, dtype=torch.bool)], dim=1)
log("loading the transformer (fp32, MPS)")
dit = QwenImage21Transformer2DModel.from_pretrained(os.path.join(args.model_dir, "transformer"),
                                                    torch_dtype=torch.float32).to(dev)
dit.eval()
probes = {}


def grab(name):
    def f(module, a, o):
        probes[name] = o[0] if isinstance(o, tuple) else o
    return f


dit.transformer_blocks[0].register_forward_pre_hook(lambda m, a, kw: probes.__setitem__("x0", a[0] if a else kw["hidden_states"]), with_kwargs=True)
for b in PROBE_BLOCKS:
    dit.transformer_blocks[b].register_forward_hook(grab(f"block{b}"))
dit.norm_out.register_forward_hook(grab("norm_out"))
text_dev = embeds[None].to(dev)
for i, t in enumerate(timesteps):
    save(f"ed_step{i}_latents", latents[0])
    x = torch.cat([ref_rows, latents], dim=1).to(dev)
    t0 = time.time()
    with torch.no_grad():
        pred = dit(hidden_states=x, timestep=(t.expand(1) / 1000).to(dev), encoder_hidden_states=text_dev,
                   encoder_hidden_states_mask=None, img_shapes=img_shapes, img_mask=img_mask.to(dev),
                   return_dict=False)[0]
    torch.mps.synchronize()
    log(f"step {i}: forward {time.time() - t0:.1f} s, t {float(t):.3f}, output {tuple(pred.shape)}")
    save(f"ed_s{i}_x0", probes["x0"][0])
    for b in PROBE_BLOCKS:
        save(f"ed_s{i}_block{b}", probes[f"block{b}"][0])
    save(f"ed_s{i}_norm_out", probes["norm_out"][0])
    target = pred[:, -N:].float().cpu()
    save(f"ed_step{i}_pred", target[0])
    latents = sched.step(target, t, latents, return_dict=False)[0]
    manifest["ed_joint_rows"] = int(probes["x0"].shape[1])
save("ed_final_latents", latents[0])
write_manifest()
log("done")
