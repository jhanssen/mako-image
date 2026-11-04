#pragma once

#include "image/engine.hpp"

#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace mako::server {

// An image model as a directory: its files (the directory, or where its mako.json points) and the settings mako.json
// gives. mako.json (every field optional, paths relative to its directory):
//   {"model": DIR, "transformer": PATH, "text_encoder": PATH, "adapter": DIR, "control": FILE, "lora_dir": DIR,
//    "loras": {NAME: SCALE or {"scale": SCALE, "merge": BOOL}, ...}, "steps": N, "terminal": F, "guidance": F, "sampler": "euler" | "euler_a" | "er_sde", "skip_blocks": [N, ...],
//    "description": TEXT, "prompting": TEXT}
// LoRAs are named by file name without .safetensors, from lora_dir (default: the directory's loras/); "loras" are the
// ones merged for requests that name none.
struct ImageModel {
    std::string name, dir;
    std::string family, description; // which model it is (image::Engine::family); what it is for, from mako.json
    std::string prompting;           // how to write its prompts, from mako.json
    image::Options options;
    std::string lora_dir; // empty: none
    image::Loras loras;   // files, scales and merge choices
    uint32_t steps = 0;   // 0: the model's
    std::optional<float> terminal, guidance;
    std::optional<image::Sampler> sampler; // unset: the model's
    std::optional<std::vector<uint32_t>> skip_blocks; // the negative prompt's pass's; unset: the model's
    // What decides the loaded weights besides LoRAs: two models with the same key load the same.
    std::string key() const
    {
        return dir + "|" + options.text_encoder + "|" + options.transformer + "|" + options.adapter + "|" +
               options.control;
    }
};

// A "skip_blocks" value (JSON: an array of block numbers); nullopt when it is not one.
std::optional<std::vector<uint32_t>> parse_skip_blocks(std::string_view json);
// The model in `dir` under `name`. Throws std::runtime_error naming the problem.
ImageModel load_image_model(const std::string& dir, const std::string& name);
// The subdirectories of `root` that hold a model or a mako.json, by name (read now); ones that fail to load are left
// out with a warning.
std::vector<ImageModel> image_models(const std::string& root);

// An image LoRA a request can name: its file name without .safetensors, its path, and for one given at start-up the
// scale it is added at by default and whether it is merged.
struct LoraChoice {
    std::string name, path;
    std::optional<float> scale;
    std::optional<bool> merge;
};
// The start-up LoRAs (in their order), then the .safetensors files in `dir` by name (read now; empty: none). A
// start-up LoRA's name wins over a file of the same name in `dir`.
std::vector<LoraChoice> lora_choices(const image::Loras& defaults, const std::string& dir);
// What a LoRA is for: its trigger (words a prompt needs for it to act), a description and the scales it is meant for.
// Each from NAME.json beside the file ({"trigger", "description", "scale_range": [low, high]}), else the trigger and
// description from the file's metadata (trigger_word or modelspec.trigger_phrase; modelspec.description); empty when
// neither has it.
struct LoraNotes {
    std::string trigger, description;
    std::optional<std::pair<float, float>> scale_range;
};
LoraNotes lora_notes(const std::string& path);
// A request's "loras" value (JSON: an array of {"name" (with or without .safetensors), "scale" (default 1)}) as files
// and scales. Throws std::invalid_argument naming the problem.
image::Loras resolve_loras(std::string_view json, const std::vector<LoraChoice>& choices);

// An ESRGAN upscaler a request can name: its file name without .safetensors, its path, what it multiplies a side by
// and its blocks (from the file's header), and what it is for, from NAME.json beside it ({"description": "..."}).
struct UpscalerChoice {
    std::string name, path, description;
    uint32_t factor = 1, blocks = 0;
};
// The .safetensors files in `dir` by name (read now; empty: none); ones that are not ESRGAN upscalers in
// Real-ESRGAN's names (tools/esrgan_to_safetensors.py makes those) are left out with a warning.
std::vector<UpscalerChoice> upscaler_choices(const std::string& dir);

} // namespace mako::server
