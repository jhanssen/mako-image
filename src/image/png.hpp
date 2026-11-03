#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace mako::image {

// RGBA [height][width][4] in [-1, 1] to 8-bit samples as the reference pipeline makes them (x / 2 + 1/2, clamped,
// times 255, rounded).
std::vector<uint8_t> to_rgba8(std::span<const float> rgba);
// A PNG of 8-bit RGBA (with `rgb`, the alpha channel left out), or a JPEG (no alpha; quality 0.92). Throws
// std::runtime_error.
std::vector<uint8_t> encode_png(std::span<const uint8_t> rgba8, uint32_t width, uint32_t height, bool rgb = false,
                                bool jpeg = false);
void write_file(const std::string& path, std::span<const uint8_t> data);
// An encoded image (PNG, JPEG, HEIC, ..., what ImageIO reads; its orientation applied): its width and height.
// Throws std::runtime_error on data that is not an image.
std::pair<uint32_t, uint32_t> image_size(std::span<const uint8_t> data);
// The image scaled to cover width x height and centred (the overhang cut off), transparency over white: RGBA
// [height][width][4] in [-1, 1].
std::vector<float> decode_image(std::span<const uint8_t> data, uint32_t width, uint32_t height);
// The size to render an image of w x h pixels to start from when none is asked for: its own, scaled down to 1024x1024
// pixels' area when larger, in multiples of 32 (at least 256 a side).
std::pair<uint32_t, uint32_t> start_size(uint32_t w, uint32_t h);

// How an image was made, for the PNG's text chunks.
struct Provenance {
    std::string prompt, model;
    std::optional<std::string> negative; // when guided
    std::string sampler = "euler"; // the API's name: "euler", "euler_a", "er_sde"
    std::vector<uint32_t> skip_blocks; // the blocks the negative prompt's pass left out
    float guidance = 1;
    uint32_t steps = 0, width = 0, height = 0;
    uint64_t seed = 0;
    std::optional<float> strength; // image to image: the share of the steps run on the starting image
    struct Lora {
        std::string name;
        float scale = 1;
        std::optional<bool> merge; // as asked (unset: the model's way)
    };
    std::vector<Lora> loras;
};
// The PNG with UTF-8 text chunks before its end: "parameters" (the prompt, then "Negative prompt: ...", then
// "Steps: ..., Sampler: ..., CFG scale: ..., Seed: ..., Size: WxH, Model: ...", as image tools commonly read it)
// and "mako" (the same as JSON, with the LoRAs). Throws std::runtime_error on data that is not a PNG.
std::vector<uint8_t> with_provenance(std::vector<uint8_t> png, const Provenance& p);

} // namespace mako::image
