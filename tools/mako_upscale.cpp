// Upscales an image with an ESRGAN model (the original ESRGAN's or Real-ESRGAN's RRDBNet; tools/esrgan_to_safetensors.py
// converts a .pth checkpoint).
//   mako_upscale <model.safetensors> <image> [-o out.png] [--scale F] [--tile N] [--overlap N] [--rgb]
// --scale: the output's size as a multiple of the input's, at most the model's factor (default: the model's, 4 for
// most; less is resampled down from the model's output with Lanczos). --tile: an image over N pixels a side runs in
// tiles of N, each with --overlap pixels of its neighbours as context (default 1024 and 32; 0: all at once).
#include "app/cli.hpp"
#include "image/esrgan.hpp"
#include "image/png.hpp"
#include "log/log.hpp"
#include "mtl/context.hpp"

#include <chrono>
#include <cmath>
#include <format>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    std::string out = "image.png";
    float scale = 0;
    uint32_t tile = 1024, overlap = 32;
    bool rgb = false;
    mako::app::Cli cli("mako_upscale", "<model.safetensors> <image>", 2, 2);
    cli.option("-o", "FILE", "the PNG to write (default image.png)", out)
        .option("--scale", "F", "the output's size as a multiple of the input's (default: the model's)", scale)
        .option("--tile", "N", "run images over N pixels a side in tiles of N (default 1024; 0: all at once)", tile)
        .option("--overlap", "N", "pixels of context around a tile (default 32)", overlap)
        .flag("--rgb", "leave the alpha channel out", rgb);
    cli.env_prefix("MAKO_UPSCALE_");
    const auto pos = cli.parse(argc, argv);
    using Clock = std::chrono::steady_clock;
    auto s_since = [](Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); };
    try {
        std::ifstream f(pos[1], std::ios::binary);
        const std::vector<uint8_t> bytes{std::istreambuf_iterator<char>(f), {}};
        if (bytes.empty())
            throw std::runtime_error("cannot read " + pos[1]);
        const auto [w, h] = mako::image::image_size(bytes);
        const auto rgba = mako::image::decode_image(bytes, w, h);
        const auto t_load = Clock::now();
        mako::mtl::Context ctx;
        mako::image::Esrgan model(pos[0], ctx);
        const uint32_t factor = model.factor();
        if (scale == 0)
            scale = float(factor);
        if (!(scale >= 1) || scale > float(factor))
            throw std::runtime_error(std::format("--scale: 1 to the model's {}", factor));
        mako::log::info("{}: {} blocks, {}x, loaded in {:.1f} s", pos[0], model.blocks(), factor, s_since(t_load));
        const auto t0 = Clock::now();
        auto big = model.upscale(rgba, w, h, tile, overlap);
        const double up = s_since(t0);
        const uint32_t W = uint32_t(std::lround(w * scale)), H = uint32_t(std::lround(h * scale));
        if (W != w * factor || H != h * factor)
            big = mako::image::resize_image(big, w * factor, h * factor, W, H);
        mako::image::write_file(out, mako::image::encode_png(mako::image::to_rgba8(big), W, H, rgb));
        mako::log::info("{}x{} -> {}x{} in {:.1f} s ({:.1f} s upscaling: {:.1f} s on the GPU, {:.1f} s encoding): {}", w, h,
                        W, H, s_since(t0), up, model.stats().gpu_exec_ms / 1e3, model.stats().encode_ms / 1e3, out);
    } catch (const std::exception& e) {
        mako::log::error("{}", e.what());
        return 1;
    }
    return 0;
}
