// ESRGAN upscalers against PyTorch's output (fp32 on the CPU, spandrel's RRDBNet) for a 48x32 crop of a render: for
// each directory NAME of the reference directory (manifest.json {"model", "width", "height", "factor"}, input.f32 and
// output.f32: RGBA channels-last in [-1, 1]), the model NAME.safetensors of the upscalers directory, all at once and
// in tiles.
//   esrgan_test <upscalers dir> <reference dir> [model name]
// With MAKO_DUMP_ESRGAN=PREFIX the stages of the first model's untiled run are written to PREFIX.<stage> (f32).
#include "image/esrgan.hpp"
#include "mtl/context.hpp"

#include <glaze/glaze.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <print>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

int failures = 0;

void check(bool ok, const std::string& what)
{
    std::println("  {:4} {}", ok ? "ok" : "FAIL", what);
    failures += !ok;
}

std::string read_file(const fs::path& p)
{
    std::ifstream in(p, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

std::vector<float> read_f32(const fs::path& p)
{
    const std::string s = read_file(p);
    std::vector<float> v(s.size() / 4);
    std::memcpy(v.data(), s.data(), v.size() * 4);
    return v;
}

// Over the colour channels of RGBA pixels: relative RMS error and the largest difference.
struct Diff {
    double rel = 0, max = 0;
};
Diff diff(const std::vector<float>& got, const std::vector<float>& want)
{
    Diff d;
    double e = 0, r = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        if (i % 4 == 3)
            continue;
        e += (double(got[i]) - want[i]) * (double(got[i]) - want[i]);
        r += double(want[i]) * want[i];
        d.max = std::max(d.max, std::abs(double(got[i]) - want[i]));
    }
    d.rel = std::sqrt(e / r);
    return d;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 3 || !fs::is_directory(argv[1]) || !fs::is_directory(argv[2])) {
        std::println("upscalers {} or references {} not found: skipped", argc > 1 ? argv[1] : "",
                     argc > 2 ? argv[2] : "");
        return 77;
    }
    const fs::path models = argv[1], refs = argv[2];
    std::vector<fs::path> dirs;
    for (const auto& e : fs::directory_iterator(refs))
        if (e.is_directory() && fs::exists(e.path() / "manifest.json") &&
            (argc < 4 || e.path().filename() == argv[3]))
            dirs.push_back(e.path());
    std::ranges::sort(dirs);
    if (dirs.empty()) {
        std::println("no references under {}: skipped", refs.string());
        return 77;
    }
    mako::mtl::Context ctx;
    for (const fs::path& ref : dirs) {
        const std::string name = ref.filename().string();
        std::println("{}", name);
        glz::generic manifest;
        if (glz::read_json(manifest, read_file(ref / "manifest.json"))) {
            check(false, "manifest.json");
            continue;
        }
        const uint32_t w = uint32_t(manifest["width"].get_number()), h = uint32_t(manifest["height"].get_number());
        const uint32_t factor = uint32_t(manifest["factor"].get_number());
        const fs::path file = models / (name + ".safetensors");
        if (!fs::exists(file)) {
            check(false, std::format("{} is missing", file.string()));
            continue;
        }
        try {
            const auto t_load = std::chrono::steady_clock::now();
            mako::image::Esrgan model(file.string(), ctx);
            const double load_ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_load).count();
            check(model.factor() == factor,
                  std::format("{} blocks, {}x (the reference {}x), loaded in {:.0f} ms", model.blocks(),
                              model.factor(), factor, load_ms));
            const auto in = read_f32(ref / "input.f32"), want = read_f32(ref / "output.f32");
            if (in.size() != uint64_t(w) * h * 4 || want.size() != uint64_t(w) * h * factor * factor * 4) {
                check(false, "the reference's sizes");
                continue;
            }
            if (const char* dump = std::getenv("MAKO_DUMP_ESRGAN"))
                model.set_probe([dump](const std::string& stage, int, const float* data, size_t n) {
                    std::ofstream(std::format("{}.{}", dump, stage), std::ios::binary)
                        .write(reinterpret_cast<const char*>(data), std::streamsize(n * 4));
                });
            const auto t0 = std::chrono::steady_clock::now();
            const auto got = model.upscale(in, w, h, 0);
            model.set_probe(nullptr);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            // F16 weights and half unfolded activations against the fp32 reference: measured 0.8-2.2e-4 relative,
            // 1-4e-3 at the worst value (of 2 across [-1, 1]).
            Diff d = diff(got, want);
            check(d.rel < 1e-3 && d.max < 0.02,
                  std::format("{}x{} -> {}x{} all at once: rel error {:.2e}, largest {:.2e} ({:.0f} ms)", w, h,
                              w * factor, h * factor, d.rel, d.max, ms));
            // Tiles of 16 with 8 pixels of context, far less than the network's reach: the seams cost 1.5-8.8e-3
            // relative (up to 0.13 at a value); the default 1024-pixel tiles with 32 of context see far fewer seams.
            const auto tiled = model.upscale(in, w, h, 16, 8);
            d = diff(tiled, want);
            check(d.rel < 0.02,
                  std::format("in 16-pixel tiles with 8 of context: rel error {:.2e}, largest {:.2e}", d.rel, d.max));
        } catch (const std::exception& e) {
            check(false, e.what());
        }
    }
    if (failures)
        std::println("{} failed", failures);
    return failures ? 1 : 0;
}
