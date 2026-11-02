// Text to image with Qwen-Image 2.1, Krea 2, MOSS-Image, Anima or SDXL from their original files.
//   mako_image <model dir> <prompt> [-o out.png] [--width N] [--height N] [--steps N] [--seed N] [--count N] [--rgb]
//              [--lora FILE[:SCALE] ...] [--terminal F] [--text-encoder DIR] [--transformer PATH] [--adapter DIR]
//              [--negative TEXT] [--guidance F] [--sampler NAME] [--skip-blocks LIST] [--llm URL] [--llm-api-key KEY]
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
// --llm: a mako_serve on this machine whose memory to borrow while the model is loaded (it unloads its LLM meanwhile).
#include "app/cli.hpp"
#include "image/engine.hpp"
#include "image/png.hpp"
#include "log/log.hpp"
#include "memory/borrower.hpp"
#include "mtl/context.hpp"

#include <Foundation/Foundation.hpp>

#include <chrono>
#include <filesystem>
#include <format>
#include <optional>
#include <print>
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
    uint32_t width = 1024, height = 1024, steps = 0;
    uint64_t seed = 0;
    uint32_t count = 1;
    mako::image::Loras loras;
    std::optional<float> terminal, guidance;
    std::optional<std::string> negative;
    std::optional<mako::image::Sampler> sampler;
    std::optional<std::vector<uint32_t>> skip_blocks;
    std::string text_encoder, transformer, adapter;
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
        .option("--terminal", "F", "the schedule's last noise level (0: none)",
                [&](const std::string& v) { terminal = mako::app::Cli::parse<float>(v); })
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
        auto model = mako::image::Engine::open(pos[0], ctx, {text_encoder, transformer, adapter});
        auto& engine = *model;
        for (const auto& l : loras)
            engine.add_lora(l);
        mako::log::info("loaded in {:.1f} s", ms_since(t_load) / 1e3);
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
            auto t_step = Clock::now();
            r.progress = [&](uint32_t i, uint32_t total) {
                mako::log::info("step {} of {}: {:.0f} ms, thermal {}", i, total, ms_since(t_step),
                                thermal_state());
                t_step = Clock::now();
            };
            const auto t0 = Clock::now();
            const auto img = engine.generate(r);
            const double gen = ms_since(t0);
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
            mako::image::write_file(path, mako::image::with_provenance(
                                              mako::image::encode_png(mako::image::to_rgba8(img), width, height, rgb),
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
