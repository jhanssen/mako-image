#!/usr/bin/env python3
# Reference tensors for mako-image's MiniMax-H3-Image tests, from the Diffusers implementation at a small canvas:
#   PYTHONPATH=<a diffusers checkout with MiniMax-H3-Image> python tools/dump_h3_image_ref.py <model dir> <out dir>
#                                                           [--stage text|dit|vae|edit|all]
# <model dir> holds the release's files (transformer/ shards with pdd_fused_heads.safetensors, vae_diffusers/,
# tokenizer/, processor/) and MiniMax-H3's text_encoder/ (a link will do).
# Writes manifest.json and raw float32 files (row-major, shapes in the manifest), in the layouts mako uses: packed
# rows [patches][96] (a 2x2 patch's [channel][dy][dx] values), latents [h][w][24], pixels [H][W][3]. The text
# encoder runs in fp32 on the CPU (each layer upcast for its forward), the transformer on MPS with its BF16 weights
# upcast to fp32 per module (what mako computes: fp32 activations, BF16 weights), the VAE in fp32 on MPS. Every stage
# reads the earlier ones' files, so a stage can be rerun alone.
import gc
import json
import math
import os
import sys
import time

import numpy as np
import torch

PROMPT = ("A photograph of a red fox sitting in fresh snow in a pine forest at sunrise, soft golden light on its fur, "
          "its breath visible in the cold air, pine branches heavy with snow behind it.")
EDIT_PROMPT = ("Keep the scene of <Picture 1> as it is, but put the yellow disc of <Picture 2> in the sky as a sun. "
               "A smooth gradient landscape with a single disc and a bar, as a flat illustration.")
SIZE = 256         # a 256x256 canvas: 16x16 latents, 8x8 = 64 transformer rows
SEED = 42
TEXT_LAYER = 50
PROBE_BLOCKS = (0, 1, 24, 49)
EDIT_STEPS = 2     # the edit stage runs the first two steps (heads 0 and 1)
TOKEN_CASES = ["", "Hello, world!", PROMPT, "<Picture 1> and <Picture 2>", "日本語のテスト 123 456",
               "émigré café ÄÖÜ ß", "tabs\tand\nnewlines  double  spaces",
               "<|vision_start|><|image_pad|><|vision_end|> text after", "😀 emoji 🎬"]

model_dir, out = sys.argv[1], sys.argv[2]
stage = sys.argv[4] if len(sys.argv) > 4 and sys.argv[3] == "--stage" else "all"
os.makedirs(out, exist_ok=True)
manifest_path = os.path.join(out, "manifest.json")
manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {"tensors": {}}


def save(name, t):
    t = t.detach().float().contiguous().cpu()
    t.numpy().tofile(os.path.join(out, name + ".f32"))
    manifest["tensors"][name] = list(t.shape)


def save_f64(name, t):
    t = t.detach().double().contiguous().cpu()
    t.numpy().tofile(os.path.join(out, name + ".f64"))
    manifest["tensors"][name] = list(t.shape)


def load(name):
    return torch.from_numpy(np.fromfile(os.path.join(out, name + ".f32"), dtype=np.float32).reshape(
        manifest["tensors"][name]))


def write_manifest():
    json.dump(manifest, open(manifest_path, "w"), indent=1)


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def patchify(latents):
    """[1][24][h][w] -> rows [h/2 * w/2][96], a patch's [c][dy][dx] values (the pipeline's patchify)."""
    from diffusers.modular_pipelines.minimax_h3_image.before_denoise import patchify_video_latents
    return patchify_video_latents(latents[:, :, None], (1, 2, 2))


def unpatchify(rows, h, w):
    """rows [h/2 * w/2][96] -> [1][24][h][w]."""
    r = rows.reshape(1, h // 2, w // 2, 24, 2, 2).permute(0, 3, 1, 4, 2, 5)
    return r.reshape(1, 24, h, w).contiguous()


def hwc(latents):
    """[1][24][h][w] -> [h][w][24]."""
    return latents[0].permute(1, 2, 0).contiguous()


def synth(w, h, seed):
    """A smooth test picture: colour gradients, a disc and a bar, so resizing and encoding have structure."""
    from PIL import Image
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    rng = np.random.default_rng(seed)
    cx, cy, r = rng.uniform(0.3, 0.7) * w, rng.uniform(0.3, 0.7) * h, 0.2 * min(w, h)
    img = np.stack([xx / w, yy / h, 0.5 + 0.5 * np.sin(6.28 * (xx / w + yy / h))], axis=-1)
    disc = (xx - cx) ** 2 + (yy - cy) ** 2 < r * r
    img[disc] = rng.uniform(0, 1, 3)
    img[int(0.1 * h):int(0.15 * h), int(0.2 * w):int(0.8 * w)] = (0.9, 0.9, 0.1)
    return Image.fromarray(np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8), "RGB")


def load_text_encoder(vision=False):
    """The text encoder (BF16, CPU), each layer upcast to fp32 for its forward; the vision tower in fp32."""
    from transformers import Qwen3VLForConditionalGeneration

    te = Qwen3VLForConditionalGeneration.from_pretrained(os.path.join(model_dir, "text_encoder"), dtype=torch.bfloat16,
                                                         device_map="cpu")
    lm = te.model.language_model
    # hidden_states[50] is the output of 50 layers before any norm: the last kept layer's output would come back
    # normed, so keep 51.
    lm.layers = lm.layers[:TEXT_LAYER + 1]
    lm.config.num_hidden_layers = TEXT_LAYER + 1
    te.config.text_config.num_hidden_layers = TEXT_LAYER + 1
    gc.collect()
    lm.embed_tokens.float()
    lm.norm.float()
    lm.rotary_emb.float()
    if vision:
        te.model.visual.float()

    def upcast_pre(module, args, kwargs):
        module.float()
        return args, kwargs

    def downcast_post(module, args, output):
        module.to(torch.bfloat16)
        return output

    for layer in lm.layers:
        layer.register_forward_pre_hook(upcast_pre, with_kwargs=True)
        layer.register_forward_hook(downcast_post)
    return te


# ---------------------------------------------------------------------------------------------------- text encoder
if stage in ("text", "all"):
    from transformers import Qwen2TokenizerFast

    tok = Qwen2TokenizerFast.from_pretrained(os.path.join(model_dir, "tokenizer"))
    ids = tok(PROMPT, add_special_tokens=False)["input_ids"]
    manifest["prompt"] = PROMPT
    manifest["ids"] = ids
    manifest["token_cases"] = [{"text": t, "ids": tok(t, add_special_tokens=False)["input_ids"]} for t in TOKEN_CASES]
    log(f"{len(ids)} prompt tokens")
    log("loading the text encoder (BF16, CPU)")
    te = load_text_encoder()
    input_ids = torch.tensor([ids], dtype=torch.long)
    log("encoding in fp32")
    with torch.no_grad():
        o = te.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                     mm_token_type_ids=torch.zeros_like(input_ids), use_cache=False, output_hidden_states=True)
    for i in (0, 1, TEXT_LAYER):
        save(f"text_hidden{i}", o.hidden_states[i][0])
    emb = o.hidden_states[TEXT_LAYER][0]
    save("text_embeds", emb)
    log("text_embeds", tuple(emb.shape), "rms", float(emb.float().pow(2).mean().sqrt()))
    del te, o
    gc.collect()
    write_manifest()


# ----------------------------------------------------------------------------------------------------- transformer
def load_dit(dev):
    """The release's shards (BF16, the patch projection and time MLP F32) and per-step heads into the diffusers model
    on MPS, every BF16 leaf upcast to fp32 for its forward (its float inputs too)."""
    from safetensors import safe_open
    from diffusers import MiniMaxH3ImageTransformer2DModel

    tdir = os.path.join(model_dir, "transformer")
    log(f"loading the transformer (BF16 shards onto MPS) from {tdir}")
    heads, head_dim = 56, 128
    sd = {}
    for name in sorted(f for f in os.listdir(tdir) if f.endswith(".safetensors")):
        with safe_open(os.path.join(tdir, name), framework="pt", device="cpu") as f:
            for k in f.keys():
                t = f.get_tensor(k)
                if k == "rope.inv_freq" or k.startswith("final_layer.video_out."):
                    continue  # the per-step heads replace the base head
                tk = k.replace("token_refiner.blocks.", "token_refiner.refiner_blocks.", 1) if k.startswith(
                    "token_refiner.blocks.") else k.replace("blocks.", "transformer_blocks.", 1) if k.startswith("blocks.") else k
                for a, b in (("time_embedder.proj_in.", "time_embedder.linear_1."),
                             ("time_embedder.proj_out.", "time_embedder.linear_2."),
                             ("video_patch_proj.", "proj_in."), ("condition_proj.", "context_embedder."),
                             ("final_layer.norm.", "norm_out.norm."),
                             ("final_layer.adaln_proj.linear.", "norm_out.linear."),
                             (".attn.q_norm.", ".attn.norm_q."), (".attn.k_norm.", ".attn.norm_k."),
                             (".attn.out_proj.", ".attn.to_out.0."), (".mlp.fc2.", ".ff.net.2.")):
                    tk = tk.replace(a, b)
                if tk.endswith(".attn.qkv_proj.weight"):
                    # Rows interleaved per head [q k v] in the shards; q, k and v apart.
                    g = t.reshape(heads, 3 * head_dim, -1)
                    q, kk, v = (x.reshape(heads * head_dim, -1).to(dev) for x in g.split(head_dim, dim=1))
                    p = tk.removesuffix("qkv_proj.weight")
                    sd[p + "to_q.weight"], sd[p + "to_k.weight"], sd[p + "to_v.weight"] = q, kk, v
                elif tk.endswith(".mlp.fc1.weight"):
                    # [gate; value] in the checkpoint; diffusers' SwiGLU takes [value; gate].
                    gate, value = t.chunk(2, dim=0)
                    sd[tk.replace(".mlp.fc1.weight", ".ff.net.0.proj.weight")] = torch.cat([value, gate]).to(dev)
                else:
                    sd[tk] = t.to(dev)
    with safe_open(os.path.join(model_dir, "pdd_fused_heads.safetensors"), framework="pt", device="cpu") as f:
        hw, hb = f.get_tensor("video_out.weight"), f.get_tensor("video_out.bias")
    for s in range(hw.shape[0]):
        sd[f"pdd_proj_out.{s}.weight"] = hw[s].contiguous().to(dev)
        sd[f"pdd_proj_out.{s}.bias"] = hb[s].contiguous().to(dev)
    with torch.device("meta"):
        dit = MiniMaxH3ImageTransformer2DModel()
    missing, unexpected = dit.load_state_dict(sd, strict=False, assign=True)
    missing = [k for k in missing if not k.endswith("rope.inv_freq")]
    if missing or unexpected:
        raise RuntimeError(f"state dict mismatch: missing {missing[:5]}, unexpected {unexpected[:5]}")
    dit.rope.inv_freq = 1.0 / (10000.0 ** (torch.arange(0, 32, 2, dtype=torch.float32, device=dev) / 32))
    dit.eval()
    del sd
    gc.collect()
    dtypes = {}
    for n, p in dit.named_parameters():
        dtypes[str(p.dtype)] = dtypes.get(str(p.dtype), 0) + p.numel()
    log("parameters by dtype", dtypes)

    def pre(module, args, kwargs):
        module.float()
        args = tuple(a.float() if torch.is_tensor(a) and a.is_floating_point() else a for a in args)
        kwargs = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in kwargs.items()}
        return args, kwargs

    def post(module, args, output):
        module.to(torch.bfloat16)
        return output

    for name, m in dit.named_modules():
        ps = list(m.parameters(recurse=False))
        if ps and all(p.dtype == torch.bfloat16 for p in ps):
            m.register_forward_pre_hook(pre, with_kwargs=True)
            m.register_forward_hook(post)
    return dit


def schedule():
    """The checkpoint's grid as the pipeline sets it: every sigma rounded through its float timestep 1 - sigma."""
    from diffusers import MiniMaxH3Scheduler
    grid = json.load(open(os.path.join(model_dir, "pdd_config.json")))["sigmas"]
    timesteps = torch.tensor([1.0 - s for s in grid[:-1]], dtype=torch.float32)
    sigmas = torch.cat([1.0 - timesteps, torch.zeros(1)])
    sched = MiniMaxH3Scheduler(shift_provided_sigmas=False, euler_update_form="velocity")
    sched.set_timesteps(sigmas=sigmas)
    manifest["grid"] = grid
    manifest["sigmas"] = sched.sigmas.tolist()
    manifest["timesteps"] = sched.timesteps.tolist()
    return sched


def run_dit(prefix, text_embeds, text_tags, reference_latents, reference_noise, steps):
    """`steps` steps of the transformer over [text | references | image] from the seed's noise; the layout, noise,
    per-step rows, probes and velocities saved under `prefix`. reference_latents: the normalized [1][24][h][w]
    latents of each reference, reference_noise the noise each is mixed with at t = 0.999."""
    from diffusers.modular_pipelines.minimax_h3_image.before_denoise import build_image_packed_sequence

    dev = torch.device("mps")
    T = text_embeds.shape[0]
    text_embeds = text_embeds.to(dev, torch.bfloat16)
    lat_h = lat_w = SIZE // 16
    grids = [tuple(z.shape[-2:]) for z in reference_latents]
    pos, tags, iidx, tidx, n_ref = build_image_packed_sequence(torch.tensor(text_tags, dtype=torch.long), grids,
                                                                lat_h, lat_w, (1, 2, 2), 0)
    S = pos.shape[0]
    manifest.update({"size": SIZE, "latent_height": lat_h, "latent_width": lat_w, "seed": SEED,
                     "text_layer": TEXT_LAYER})
    manifest.update({prefix + "text_rows": T, prefix + "seq_len": S, prefix + "reference_rows": n_ref,
                     prefix + "steps": steps})
    save_f64(prefix + "position_ids", pos)
    manifest[prefix + "token_tags"] = tags.tolist()
    log(f"sequence {S} rows: {T} text, {n_ref} reference, {iidx.numel() - n_ref} image")
    sched = schedule()
    g = torch.Generator().manual_seed(SEED)
    latents = torch.randn((1, 24, lat_h, lat_w), generator=g, dtype=torch.float32)
    rows = patchify(latents)
    save(prefix + "noise_rows", rows)
    if reference_latents:
        cond = []
        for z, noise in zip(reference_latents, reference_noise):
            cond.append(patchify(sched.scale_noise(z, torch.tensor(0.999), noise)))
        cond = torch.cat(cond)
        assert cond.shape[0] == n_ref
        save(prefix + "reference_rows", cond)
        rows = torch.cat([cond, rows])
    dit = load_dit(dev)
    probes = {}

    def grab(name):
        def f(module, args, output):
            probes[name] = output
        return f

    dit.token_refiner.register_forward_hook(grab("txt"))
    dit.time_embedder.register_forward_hook(grab("temb"))
    dit.transformer_blocks[0].register_forward_pre_hook(lambda m, a: probes.__setitem__("x0", a[0]))
    dit.transformer_blocks[0].adaln_proj.register_forward_hook(grab("mod0"))
    for b in PROBE_BLOCKS:
        dit.transformer_blocks[b].register_forward_hook(grab(f"block{b}"))
    dit.norm_out.register_forward_hook(grab("norm_out"))
    for i, t in enumerate(sched.timesteps[:steps]):
        # Every row at the step's level, the references held at 0.999.
        row_t = torch.full((S,), float(t), dtype=torch.float32)
        row_t[iidx[:n_ref]] = 0.999
        uniq, tsi = torch.unique(row_t, sorted=True, return_inverse=True)
        save(f"{prefix}step{i}_rows", rows)
        t0 = time.time()
        with torch.no_grad():
            pred = dit(hidden_states=rows[None].to(dev), encoder_hidden_states=text_embeds[None],
                       timestep=uniq.to(dev), timestep_indices=tsi.to(dev), token_tags=tags.to(dev),
                       position_ids=pos.to(dev, torch.float32), image_indices=iidx.to(dev), text_indices=tidx.to(dev),
                       pdd_step_index=i, return_dict=False)[0]
        torch.mps.synchronize()
        log(f"step {i}: forward {time.time() - t0:.1f} s, timesteps {uniq.tolist()}")
        if i == 0:
            save(f"{prefix}s0_txt", probes["txt"][0])
        save(f"{prefix}s{i}_temb", probes["temb"])
        save(f"{prefix}s{i}_x0", probes["x0"][0])
        save(f"{prefix}s{i}_mod0", torch.stack(list(probes["mod0"])))  # [6][timesteps * 3][5376]
        for b in PROBE_BLOCKS:
            save(f"{prefix}s{i}_block{b}", probes[f"block{b}"][0])
        save(f"{prefix}s{i}_norm_out", probes["norm_out"][0])
        save(f"{prefix}step{i}_pred", pred[0])
        # Only the generated rows step; the references ride through.
        rows[n_ref:] = sched.step(pred[0, n_ref:].float().cpu(), t, rows[n_ref:], return_dict=False)[0]
    save(prefix + "final_rows", rows)
    del dit
    gc.collect()
    torch.mps.empty_cache()
    write_manifest()


if stage in ("dit", "all"):
    run_dit("", load("text_embeds"), [1] * len(manifest["ids"]), [], [], len(manifest.get("grid", [0] * 7)) - 1)

# ------------------------------------------------------------------------------------------------------------- VAE
if stage in ("vae", "all"):
    from diffusers import AutoencoderKLMiniMaxH3Image
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

    dev = torch.device("mps")
    lat_h, lat_w = manifest["latent_height"], manifest["latent_width"]
    vdir = os.path.join(model_dir, "vae_diffusers")
    log("loading the VAE (fp32, MPS)")
    vae = AutoencoderKLMiniMaxH3Image.from_pretrained(vdir).to(dev)
    lmean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1)
    lstd = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1)
    pm = torch.tensor((0.485, 0.456, 0.406)).view(1, -1, 1, 1)
    ps = torch.tensor((0.229, 0.224, 0.225)).view(1, -1, 1, 1)
    manifest["latents_mean"], manifest["latents_std"] = vae.config.latents_mean, vae.config.latents_std
    # The final latents decoded (tiled, as the model ships), the pipeline's 8-bit truncation after.
    final = unpatchify(load("final_rows")[manifest["reference_rows"]:], lat_h, lat_w)
    save("vae_in", hwc(final * lstd + lmean))
    t0 = time.time()
    with torch.no_grad():
        dec = vae.decode((final * lstd + lmean).to(dev), return_dict=False)[0]
    torch.mps.synchronize()
    log(f"decoded {tuple(dec.shape)} in {time.time() - t0:.1f} s")
    img = (dec.float().cpu() * ps + pm).clamp(0, 1)
    save("vae_out", img[0].permute(1, 2, 0))  # [H][W][3] in [0, 1]
    u8 = (img * 255).clamp(0, 255).to(torch.uint8)
    u8[0].permute(1, 2, 0).contiguous().numpy().tofile(os.path.join(out, "vae_out.u8"))
    from PIL import Image
    Image.fromarray(u8[0].permute(1, 2, 0).numpy(), "RGB").save(os.path.join(out, "vae_out.png"))
    # A synthetic image encoded: the posterior, and the sample with a saved noise, rounded to fp16 and normalized.
    pic = synth(SIZE, SIZE, 5)
    np.array(pic).tofile(os.path.join(out, "enc_in.u8"))
    pixels = torch.from_numpy(np.array(pic)).permute(2, 0, 1)[None].float() / 255.0
    with torch.no_grad():
        post = vae.encode(((pixels - pm) / ps).to(dev), return_dict=False)[0]
    mean, std = post.mean.float().cpu(), post.std.float().cpu()
    noise = torch.randn(mean.shape, generator=torch.Generator().manual_seed(7))
    z = (mean + std * noise).to(torch.float16).float()
    save("enc_mean", hwc(mean))
    save("enc_std", hwc(std))
    save("enc_noise", hwc(noise))
    save("enc_latents", hwc((z - lmean) / lstd))
    log(f"encoded: moments {tuple(mean.shape)}, std mean {float(std.mean()):.4f}")
    del vae
    gc.collect()
    torch.mps.empty_cache()
    write_manifest()

# ------------------------------------------------------------------------------------------------------------ edit
if stage in ("edit", "all"):
    from PIL import Image
    from transformers import Qwen2TokenizerFast, Qwen3VLProcessor
    from diffusers import AutoencoderKLMiniMaxH3Image
    from diffusers.modular_pipelines.minimax_h3_image.before_encoder import bound_short_edge
    from diffusers.image_processor import VaeImageProcessor

    # Two synthetic references, sized so the text encoder sees them at a 1024 short edge and the VAE within 1024 to
    # 2048 (both scaled up here), as the pipeline's resize step does.
    refs = [synth(400, 300, 11), synth(300, 400, 12)]
    proc = VaeImageProcessor(vae_scale_factor=16, do_normalize=False)
    vision_images, vae_images = [], []
    for i, r in enumerate(refs):
        np.array(r).tofile(os.path.join(out, f"ed_ref{i}.u8"))
        manifest[f"ed_ref{i}_size"] = list(r.size)
        vw, vh = bound_short_edge(*r.size, 1024, 1024, 32)
        ew, eh = bound_short_edge(*r.size, 1024, 2048, 32)
        vi = r if r.size == (vw, vh) else proc.resize(r, height=vh, width=vw)
        ei = r if r.size == (ew, eh) else proc.resize(r, height=eh, width=ew)
        vision_images.append(vi)
        vae_images.append(ei)
        np.array(vi).tofile(os.path.join(out, f"ed_vision{i}.u8"))
        np.array(ei).tofile(os.path.join(out, f"ed_vae{i}.u8"))
        manifest[f"ed_vision{i}_size"] = list(vi.size)
        manifest[f"ed_vae{i}_size"] = list(ei.size)
    # The presentation: a "Picture i: " label and a vision block per reference, then the prompt; vision rows tagged
    # as image (0), text rows 1.
    tok = Qwen2TokenizerFast.from_pretrained(os.path.join(model_dir, "tokenizer"))
    processor = Qwen3VLProcessor.from_pretrained(os.path.join(model_dir, "processor"))
    vision = processor.image_processor(images=vision_images, return_tensors="pt")
    grid = vision["image_grid_thw"]
    manifest["ed_image_grid_thw"] = grid.tolist()
    merge = processor.image_processor.merge_size ** 2
    ids, tags = [], []
    for i in range(len(refs)):
        label = tok(f"Picture {i + 1}: ", add_special_tokens=False)["input_ids"]
        block = ([tok.convert_tokens_to_ids("<|vision_start|>")]
                 + [tok.convert_tokens_to_ids("<|image_pad|>")] * (int(grid[i].prod()) // merge)
                 + [tok.convert_tokens_to_ids("<|vision_end|>")])
        ids += label + block
        tags += [1] * len(label) + [0] * len(block)
    prompt_ids = tok(EDIT_PROMPT, add_special_tokens=False)["input_ids"]
    ids += prompt_ids
    tags += [1] * len(prompt_ids)
    manifest["ed_prompt"], manifest["ed_ids"], manifest["ed_tags"] = EDIT_PROMPT, ids, tags
    mm_types = processor.create_mm_token_type_ids([ids])
    log(f"edit presentation: {len(ids)} tokens, grids {grid.tolist()}")

    log("loading the text encoder (BF16, CPU), the vision tower in fp32")
    te = load_text_encoder(vision=True)
    vis = te.model.visual
    vprobes = {}
    vis.merger.register_forward_hook(lambda m, a, o: vprobes.setdefault("merger", []).append(o))
    for i, m in enumerate(vis.deepstack_merger_list):
        m.register_forward_hook(lambda mod, a, o, i=i: vprobes.setdefault(f"deepstack{i}", []).append(o))
    input_ids = torch.tensor([ids], dtype=torch.long)
    log("encoding in fp32")
    with torch.no_grad():
        o = te.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                     mm_token_type_ids=torch.tensor(mm_types, dtype=torch.long),
                     pixel_values=vision["pixel_values"].float(), image_grid_thw=grid, use_cache=False,
                     output_hidden_states=True)
    # The tower ran once over both images: its merged tokens come out in one tensor, the images' one after another.
    save("ed_vis_merger", vprobes["merger"][0])
    for i in range(len(vis.deepstack_merger_list)):
        save(f"ed_vis_deepstack{i}", vprobes[f"deepstack{i}"][0])
    ed_text = o.hidden_states[TEXT_LAYER][0]
    save("ed_text_embeds", ed_text)
    log("ed_text_embeds", tuple(ed_text.shape))
    del te, o, vis
    gc.collect()
    write_manifest()

    # The references through the VAE: a posterior sample each (its noise saved), rounded to fp16, cropped to whole
    # patches and normalized; then the noise they are mixed with at t = 0.999.
    dev = torch.device("mps")
    vae = AutoencoderKLMiniMaxH3Image.from_pretrained(os.path.join(model_dir, "vae_diffusers")).to(dev)
    lmean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1)
    lstd = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1)
    pm = torch.tensor((0.485, 0.456, 0.406)).view(1, -1, 1, 1)
    ps = torch.tensor((0.229, 0.224, 0.225)).view(1, -1, 1, 1)
    g = torch.Generator().manual_seed(SEED + 1)
    ref_latents, ref_noise = [], []
    for i, img in enumerate(vae_images):
        pixels = torch.from_numpy(np.array(img)).permute(2, 0, 1)[None]
        pixels = pixels[..., : pixels.shape[-2] // 16 * 16, : pixels.shape[-1] // 16 * 16]
        pixels = (pixels.float() / 255.0 - pm) / ps
        with torch.no_grad():
            post = vae.encode(pixels.to(dev), return_dict=False)[0]
        mean, std = post.mean.float().cpu(), post.std.float().cpu()
        noise = torch.randn(mean.shape, generator=g)
        z = (mean + std * noise).to(torch.float16).float()
        z = z[..., : z.shape[-2] // 2 * 2, : z.shape[-1] // 2 * 2]
        zn = (z - lmean) / lstd
        mix = torch.randn(zn.shape, generator=g)
        save(f"ed_ref{i}_sample_noise", hwc(noise))
        save(f"ed_ref{i}_latents", hwc(zn))
        save(f"ed_ref{i}_mix_noise", hwc(mix))
        ref_latents.append(zn)
        ref_noise.append(mix)
        log(f"reference {i}: latents {tuple(zn.shape)}")
    del vae
    gc.collect()
    torch.mps.empty_cache()
    write_manifest()
    run_dit("ed_", ed_text, tags, ref_latents, ref_noise, EDIT_STEPS)

log("done")
