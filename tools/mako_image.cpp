// Text to image with Qwen-Image 2.1, Krea 2, MOSS-Image, Anima or SDXL from their original files.
//   mako_image <model dir> <prompt> [-o out.png] [--width N] [--height N] [--steps N] [--seed N] [--count N] [--rgb]
//              [--lora FILE[:SCALE] ...] [--terminal F] [--text-encoder DIR] [--transformer PATH] [--adapter DIR]
//              [--negative TEXT] [--guidance F] [--sampler NAME] [--skip-blocks LIST] [--image FILE] [--strength F]
//              [--control FILE] [--color-map FILE] [--depth-map FILE] [--color-end F] [--depth-end F]
//              [--upscale FILE[:SCALE]] [--llm URL] [--llm-api-key KEY]
// --lora adds a LoRA (repeatable; SCALE its strength, default 1; :merge into the weights or :unmerged beside them,
// default the model's way: unmerged for MOSS-Image). --terminal: the schedule's last
// noise level (0: not stretched there, as distilled few-step LoRAs are trained; default: the model's).
// --text-encoder: another directory of the text encoder's weights (e.g. a refusal-ablated one; default: the model's).
// --transformer: another file or directory of the transformer's weights (e.g. a checkpoint with LoRAs merged).
// --adapter: a MOSS-Image data-space distillation adapter (lora, time_embedding and dataspace .safetensors): its
// sampler and step count, no guidance.
// --negative: what the image should not be (classifier-free guidance: twice the transformer's work a step);
// --guidance its scale (default: the model's). --sampler: euler, euler_a (models that predict the noise: SDXL) or
// er_sde. --skip-blocks: the blocks the negative prompt's pass leaves out, e.g. 22,24 ("" for none; MOSS-Image).
// --image: start from this image (MOSS-Image), scaled to cover the size and centred (without --width and --height,
// its own size, at most 1024x1024 pixels' area); --strength: the share of the steps run on it (default 0.75).
// --control: a MOSS-Image control adapter; --color-map (a flat colour per subject) and --depth-map (near white) steer
// the layout, each for the first --color-end / --depth-end share of the steps (default 1).
// --upscale: an ESRGAN upscaler (tools/esrgan_to_safetensors.py) run on each image afterwards, to SCALE times the
// size (default: the upscaler's own, 4 for most; less is resampled down from its output).
// --llm: a mako_serve on this machine whose memory to borrow while the model is loaded (it unloads its LLM meanwhile).
#include "app/cli.hpp"
#include "image/engine.hpp"
#include "image/esrgan.hpp"
#include "image/png.hpp"
#include "log/log.hpp"
#include "memory/borrower.hpp"
#include "mtl/context.hpp"

#include <Foundation/Foundation.hpp>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <optional>
#include <print>
#include <tuple>
#include <utility>
#include <vector>

namespace {

std::string thermal_state()
{
    switch (NS::ProcessInfo::processInfo()->thermalState()) {
    case NS::ProcessInfoThermalStateNominal:
        return "nominal";
    case NS::ProcessInfoThermalStateFair:
        return "fair";
    case NS::ProcessInfoThermalStateSerious:
        return "serious";
    case NS::ProcessInfoThermalStateCritical:
        return "critical";
    }
    return "unknown";
}

} // namespace

int main(int argc, char** argv)
{
    std::string llm, llm_key;
    std::string out = "image.png";
    uint32_t width = 0, height = 0, steps = 0;
    std::string image, control, color_map, depth_map;
    float strength = 0.75f, color_end = 1.0f, depth_end = 1.0f;
    uint64_t seed = 0;
    uint32_t count = 1;
    mako::image::Loras loras;
    std::optional<float> terminal, guidance;
    std::optional<std::string> negative;
    std::optional<mako::image::Sampler> sampler;
    std::optional<std::vector<uint32_t>> skip_blocks;
    std::string text_encoder, transformer, adapter, upscaler;
    float upscale = 0;
    bool rgb = false;
    mako::app::Cli cli("mako_image", "<model dir> <prompt>", 2, 2);
    cli.option("-o", "FILE", "the PNG to write (default image.png)", out)
        .option("--width", "N", "pixels, a multiple of 32 (default 1024)", width)
        .option("--height", "N", "pixels, a multiple of 32 (default 1024)", height)
        .option("--steps", "N", "denoising steps (default: the model's)", steps)
        .option("--seed", "N", "the starting noise's seed (default 0)", seed)
        .option("--count", "N", "images, seeds counting up from --seed, out-N.png after the first (default 1)", count)
        .flag("--rgb", "leave the alpha channel out", rgb)
        .option("--lora", "FILE[:SCALE][:merge|:unmerged]", "add a LoRA (repeatable; default: the model's way)",
                [&](const std::string& v) { loras.push_back(mako::image::parse_lora(v)); })
        .option("--text-encoder", "DIR", "another directory of the text encoder's weights", text_encoder)
        .option("--transformer", "PATH", "another file or directory of the transformer's weights", transformer)
        .option("--adapter", "DIR", "a MOSS-Image data-space distillation adapter's directory", adapter)
        .option("--negative", "TEXT", "what the image should not be (guidance: two passes a step)",
                [&](const std::string& v) { negative = v; })
        .option("--guidance", "F",
                "the guidance scale, with --negative or a model guided by default (default: the model's)",
                [&](const std::string& v) { guidance = mako::app::Cli::parse<float>(v); })
        .option("--sampler", "NAME", "euler, euler_a (SDXL) or er_sde (default: the model's)",
                [&](const std::string& v) {
                    sampler = mako::image::parse_sampler(v);
                    if (!sampler)
                        throw std::invalid_argument("--sampler: euler, euler_a or er_sde");
                })
        .option("--skip-blocks", "LIST",
                "blocks the negative prompt's pass leaves out, e.g. 22,24 (\"\": none; MOSS-Image; default: the "
                "model's)",
                [&](const std::string& v) {
                    skip_blocks.emplace();
                    for (size_t at = 0; at < v.size();) {
                        const size_t comma = std::min(v.find(',', at), v.size());
                        skip_blocks->push_back(mako::app::Cli::parse<uint32_t>(v.substr(at, comma - at)));
                        at = comma + 1;
                    }
                })
        .option("--image", "FILE", "start from this image (MOSS-Image)", image)
        .option("--strength", "F", "with --image, the share of the steps run on it (default 0.75)", strength)
        .option("--control", "FILE", "a MOSS-Image control adapter", control)
        .option("--color-map", "FILE", "with --control: a colour region map", color_map)
        .option("--depth-map", "FILE", "with --control: a depth map, near white", depth_map)
        .option("--color-end", "F", "the share of the steps the colour map steers (default 1)", color_end)
        .option("--depth-end", "F", "the share of the steps the depth map steers (default 1)", depth_end)
        .option("--terminal", "F", "the schedule's last noise level (0: none)",
                [&](const std::string& v) { terminal = mako::app::Cli::parse<float>(v); })
        .option("--upscale", "FILE[:SCALE]", "an ESRGAN upscaler run on the images, to SCALE times the size",
                [&](const std::string& v) {
                    const size_t colon = v.rfind(':');
                    upscaler = v;
                    if (colon != std::string::npos && colon > 0 && v.find('/', colon) == std::string::npos) {
                        upscaler = v.substr(0, colon);
                        upscale = mako::app::Cli::parse<float>(v.substr(colon + 1));
                    }
                })
        .option("--llm", "URL", "a mako_serve on this machine to borrow memory from", llm)
        .option("--llm-api-key", "KEY", "the key that mako_serve wants", llm_key)
        .secret();
    cli.env_prefix("MAKO_IMAGE_");
    const auto pos = cli.parse(argc, argv);
    using Clock = std::chrono::steady_clock;
    auto ms_since = [](Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); };
    try {
        const auto t_load = Clock::now();
        std::optional<mako::memory::Borrower> borrower; // the LLM's memory, given back on the way out
        if (!llm.empty()) {
            borrower.emplace(llm, llm_key, "mako_image");
            borrower->acquire();
            mako::log::info("the LLM's memory is free ({:.1f} s)", ms_since(t_load) / 1e3);
        }
        mako::mtl::Context ctx;
        auto model = mako::image::Engine::open(pos[0], ctx, {text_encoder, transformer, adapter, control});
        auto& engine = *model;
        for (const auto& l : loras)
            engine.add_lora(l);
        std::optional<mako::image::Esrgan> up;
        if (!upscaler.empty()) {
            up.emplace(upscaler, ctx);
            if (upscale == 0)
                upscale = float(up->factor());
            if (!(upscale >= 1) || upscale > float(up->factor()))
                throw std::runtime_error(std::format("--upscale: the scale goes from 1 to the upscaler's {}", up->factor()));
        }
        mako::log::info("loaded in {:.1f} s", ms_since(t_load) / 1e3);
        std::vector<float> start;
        std::vector<uint8_t> bytes;
        if (!image.empty()) {
            std::ifstream f(image, std::ios::binary);
            bytes.assign(std::istreambuf_iterator<char>(f), {});
            if (!f && !f.eof())
                throw std::runtime_error("cannot read " + image);
            if (!width && !height) {
                const auto [w, h] = mako::image::image_size(bytes);
                std::tie(width, height) = mako::image::start_size(w, h);
            }
        }
        width = width ? width : 1024;
        height = height ? height : 1024;
        if (!image.empty())
            start = mako::image::decode_image(bytes, width, height);
        auto read_map = [&](const std::string& path) {
            if (path.empty())
                return std::vector<float>{};
            std::ifstream f(path, std::ios::binary);
            const std::vector<uint8_t> b{std::istreambuf_iterator<char>(f), {}};
            if (b.empty())
                throw std::runtime_error("cannot read " + path);
            return mako::image::decode_image(b, width, height);
        };
        const auto cmap = read_map(color_map), dmap = read_map(depth_map);
        for (uint32_t n = 0; n < count; ++n) {
            mako::image::Engine::Request r;
            r.prompt = pos[1];
            r.width = width;
            r.height = height;
            r.steps = steps;
            r.seed = seed + n;
            r.terminal = terminal;
            r.negative = negative;
            r.guidance = guidance;
            r.sampler = sampler;
            r.skip_blocks = skip_blocks;
            r.image = start;
            r.strength = strength;
            r.control = {cmap, dmap, color_end, depth_end};
            auto t_step = Clock::now();
            r.progress = [&](uint32_t i, uint32_t total) {
                mako::log::info("step {} of {}: {:.0f} ms, thermal {}", i, total, ms_since(t_step),
                                thermal_state());
                t_step = Clock::now();
            };
            const auto t0 = Clock::now();
            auto img = engine.generate(r);
            const double gen = ms_since(t0);
            uint32_t out_w = width, out_h = height;
            if (up) {
                const auto t_up = Clock::now();
                img = up->upscale(img, width, height);
                out_w = uint32_t(std::lround(width * upscale));
                out_h = uint32_t(std::lround(height * upscale));
                if (out_w != width * up->factor() || out_h != height * up->factor())
                    img = mako::image::resize_image(img, width * up->factor(), height * up->factor(), out_w, out_h);
                mako::log::info("upscaled to {}x{} in {:.1f} s", out_w, out_h, ms_since(t_up) / 1e3);
            }
            std::string path = out;
            if (n > 0) {
                const size_t dot = out.rfind('.');
                path = dot == std::string::npos ? std::format("{}-{}", out, n)
                                                : std::format("{}-{}{}", out.substr(0, dot), n, out.substr(dot));
            }
            mako::image::Provenance prov = engine.provenance(r);
            prov.model = std::filesystem::path(pos[0]).lexically_normal().filename().string();
            for (const auto& l : loras)
                prov.loras.push_back({std::filesystem::path(l.path).stem().string(), l.scale, l.merge});
            if (up)
                prov.upscale = {std::filesystem::path(upscaler).stem().string(), upscale};
            mako::image::write_file(path, mako::image::with_provenance(
                                              mako::image::encode_png(mako::image::to_rgba8(img), out_w, out_h, rgb),
                                              prov));
            mako::log::info("{}x{}, {} steps in {:.1f} s: {}", width, height, steps ? steps : engine.default_steps(),
                            gen / 1e3, path);
        }
    } catch (const std::exception& e) {
        mako::log::error("{}", e.what());
        return 1;
    }
    return 0;
}
