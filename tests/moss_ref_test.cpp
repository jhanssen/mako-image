// MOSS-Image against the reference pipeline's tensors (fp32 on the CPU, 256x256, 2 steps, guidance 3, seed 42;
// written by a script beside the model, manifest.json listing them):
//   tokens   the chat template's token ids, of the prompt and of the empty negative prompt
//   text     the text encoder's final hidden states, and three layers' outputs
//   dit      the transformer on the reference's text states and latents: stages of the first step's two passes and
//            every pass's prediction (the negative prompt's skipping a block)
//   vae      the VAE decoder on the reference's final latents
//   encode   the VAE encoder on the reference's image: its stages and the normalized, packed latents
//   pipeline the prompt to the image (the reference's starting noise)
//   lora     a made-up LoRA (rank 16 on every projection of three blocks, far above BF16's rounding) added unmerged
//            twice at 0.5, then merged at -1: the output must come back to the one without it (what is left, ~2e-3,
//            is rounding: it stays put as the LoRA grows, where a fault in the unmerged path would grow with it)
//   moss_ref_test <model dir> <reference dir> <mode>
#include "image/moss_image.hpp"
#include "mtl/context.hpp"

#include <glaze/glaze.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <map>
#include <print>
#include <random>
#include <string>
#include <vector>

#include <unistd.h>

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

// Relative RMS error over everything, and the lowest cosine similarity of a row of `width`.
struct Diff {
    double rel = 0, min_cos = 1;
};
Diff diff(const std::vector<float>& got, const std::vector<float>& want, size_t width)
{
    Diff d;
    double e = 0, r = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        e += (double(got[i]) - want[i]) * (double(got[i]) - want[i]);
        r += double(want[i]) * want[i];
    }
    d.rel = std::sqrt(e / r);
    for (size_t row = 0; row < want.size() / width; ++row) {
        double gw = 0, gg = 0, ww = 0;
        for (size_t i = row * width; i < (row + 1) * width; ++i) {
            gw += double(got[i]) * want[i];
            gg += double(got[i]) * got[i];
            ww += double(want[i]) * want[i];
        }
        d.min_cos = std::min(d.min_cos, gw / std::sqrt(gg * ww));
    }
    return d;
}

// [c][n] to [n][c] and back.
std::vector<float> transposed(const std::vector<float>& v, size_t rows, size_t cols)
{
    std::vector<float> o(v.size());
    for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < cols; ++c)
            o[c * rows + r] = v[r * cols + c];
    return o;
}

std::vector<int32_t> ids(const glz::generic& v)
{
    std::vector<int32_t> out;
    for (const auto& x : v.get_array())
        out.push_back(int32_t(x.get_number()));
    return out;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 4 || !fs::exists(fs::path(argv[1]) / "model_index.json") ||
        !fs::exists(fs::path(argv[2]) / "manifest.json")) {
        std::println("model {} or reference {} not found: skipped", argc > 1 ? argv[1] : "", argc > 2 ? argv[2] : "");
        return 77;
    }
    const fs::path model = argv[1], ref = argv[2];
    const std::string mode = argv[3];
    glz::generic manifest;
    if (glz::read_json(manifest, read_file(ref / "manifest.json"))) {
        std::println(stderr, "bad manifest");
        return 1;
    }
    std::println("{}", mode);
    const std::string prompt = manifest["prompt"].get_string();
    const uint32_t size = uint32_t(manifest["size"].get_number()), g = size / 16, N = g * g;
    auto shape = [&](const std::string& name) {
        std::vector<size_t> s;
        for (const auto& v : manifest["tensors"][name].get_array())
            s.push_back(size_t(v.get_number()));
        return s;
    };
    mako::mtl::Context ctx;
    mako::image::MossImage engine(model.string(), ctx);

    if (mode == "tokens") {
        for (const auto& [name, text] : {std::pair{"pos", prompt}, std::pair{"neg", std::string()}}) {
            const auto got = engine.tokenizer().encode(mako::image::MossImage::templated(text), true);
            const auto want = ids(manifest[std::string(name) + "_ids"]);
            check(got == want, std::format("{}: {} token ids (the reference's {})", name, got.size(), want.size()));
        }
    } else if (mode == "text") {
        const auto want_ids = ids(manifest["pos_ids"]);
        const auto got = engine.encode_text(want_ids, 0, {}, true);
        const auto want = read_f32(ref / "text.f32");
        if (got.size() != want.size()) {
            check(false, std::format("{} values, the reference {}", got.size(), want.size()));
        } else {
            const Diff d = diff(got, want, 4096);
            check(d.rel < 0.03, std::format("{} rows: rel error {:.2e}, lowest row cosine {:.6f}", got.size() / 4096,
                                            d.rel, d.min_cos));
        }
        const uint32_t taps[3] = {1, 4, 32};
        const auto layers = engine.encode_text(want_ids, 0, taps);
        const size_t S = want_ids.size();
        for (uint32_t j = 0; j < 3; ++j) {
            std::vector<float> a;
            for (size_t r = 0; r < S; ++r)
                a.insert(a.end(), layers.begin() + (r * 3 + j) * 4096, layers.begin() + (r * 3 + j + 1) * 4096);
            const Diff dt = diff(a, read_f32(ref / std::format("text_layer{}.f32", taps[j] - 1)), 4096);
            std::println("        layer {}: rel error {:.2e}, lowest row cosine {:.5f}", taps[j] - 1, dt.rel, dt.min_cos);
        }
    } else if (mode == "dit") {
        std::map<std::string, std::vector<float>> got;
        uint32_t call = 0;
        engine.set_probe([&](const std::string& name, int layer, const float* data, size_t n) {
            if (call < 2)
                got[std::format("c{}_{}{}", call, name, layer >= 0 ? std::to_string(layer) : "")].assign(data, data + n);
        });
        engine.begin(read_f32(ref / "text.f32"), g, g, 0);
        call = 1;
        engine.begin(read_f32(ref / "text_neg.f32"), g, g, 1);
        const auto& sig = manifest["sigmas"].get_array();
        const size_t calls = manifest["timesteps"].get_array().size();
        for (call = 0; call < calls; ++call) {
            const auto lat = transposed(read_f32(ref / std::format("c{}_latents.f32", call)), 128, N);
            const auto pred = engine.step(lat, float(sig[call / 2].get_number()), call % 2);
            const Diff d = diff(transposed(pred, N, 128), read_f32(ref / std::format("c{}_pred.f32", call)), N);
            check(d.rel < 0.05, std::format("step {} {} prediction: rel error {:.2e}, lowest channel cosine {:.4f}",
                                            call / 2, call % 2 ? "negative" : "positive", d.rel, d.min_cos));
        }
        for (const auto& [name, width] : std::vector<std::pair<std::string, size_t>>{
                 {"c0_text_emb", 3072}, {"c1_text_emb", 3072}, {"c0_mod", 18432}, {"c0_img_in", 3072},
                 {"c0_block0", 3072}, {"c0_block21", 3072}, {"c0_block41", 3072}, {"c1_block21", 3072},
                 {"c1_block41", 3072}}) {
            const auto want_s = read_f32(ref / (name + ".f32"));
            const auto it = got.find(name);
            if (it == got.end() || it->second.size() != want_s.size()) {
                check(false, std::format("{}: {} values, the reference {}", name,
                                         it == got.end() ? 0 : it->second.size(), want_s.size()));
                continue;
            }
            const Diff d = diff(it->second, want_s, width);
            std::println("  {:4} {}: rel error {:.2e}, lowest row cosine {:.5f}", "", name, d.rel, d.min_cos);
        }
    } else if (mode == "lora") {
        engine.begin(read_f32(ref / "text.f32"), g, g, 0);
        const auto lat = transposed(read_f32(ref / "c0_latents.f32"), 128, N);
        const float sigma = float(manifest["sigmas"].get_array()[0].get_number());
        const auto base = engine.step(lat, sigma, 0);
        // The made-up LoRA's file: F32 factors, no alpha (scale 1).
        const fs::path file = fs::temp_directory_path() / std::format("moss_lora_test_{}.safetensors", getpid());
        {
            std::mt19937 rng(7);
            std::normal_distribution<float> nd;
            const uint32_t rank = 16;
            std::string header = "{";
            std::vector<float> data;
            auto put = [&](const std::string& name, uint32_t rows, uint32_t cols, float sd) {
                header += std::format("{}\"{}\":{{\"dtype\":\"F32\",\"shape\":[{},{}],\"data_offsets\":[{},{}]}}",
                                      header.size() > 1 ? "," : "", name, rows, cols, data.size() * 4,
                                      (data.size() + uint64_t(rows) * cols) * 4);
                for (uint64_t i = 0; i < uint64_t(rows) * cols; ++i)
                    data.push_back(nd(rng) * sd);
            };
            for (uint32_t layer : {0u, 21u, 41u})
                for (const auto& [m, k, n] : {std::tuple{"q", 3072u, 3072u}, {"k", 3072u, 3072u}, {"v", 3072u, 3072u},
                                              {"o", 3072u, 3072u}, {"ffn_up_gate", 3072u, 16384u},
                                              {"ffn_down", 8192u, 3072u}}) {
                    const std::string p = std::format("diffusion_model.layers.{}.proj.{}.", layer, m);
                    put(p + "lora_down.weight", rank, k, 1.0f / std::sqrt(float(k)));
                    put(p + "lora_up.weight", n, rank, 0.3f);
                }
            header += "}";
            header.resize((header.size() + 7) / 8 * 8, ' ');
            std::ofstream out(file, std::ios::binary);
            const uint64_t hn = header.size();
            out.write(reinterpret_cast<const char*>(&hn), 8);
            out << header;
            out.write(reinterpret_cast<const char*>(data.data()), std::streamsize(data.size() * 4));
        }
        engine.add_lora(file.string(), 0.5f, false);
        engine.add_lora(file.string(), 0.5f, false);
        const auto with = engine.step(lat, sigma, 0);
        engine.add_lora(file.string(), -1.0f, true);
        const auto back = engine.step(lat, sigma, 0);
        fs::remove(file);
        double effect = 0, residual = 0, norm = 0;
        for (size_t i = 0; i < base.size(); ++i) {
            effect += double(with[i] - base[i]) * (with[i] - base[i]);
            residual += double(back[i] - base[i]) * (back[i] - base[i]);
            norm += double(base[i]) * base[i];
        }
        effect = std::sqrt(effect / norm);
        residual = std::sqrt(residual / norm);
        check(effect > 0.05 && residual < 0.05 * effect,
              std::format("the LoRA moves the output by {:.3e} (relative), unmerged then merged back by {:.3e}", effect,
                          residual));
    } else if (mode == "vae") {
        std::map<std::string, std::vector<float>> stages;
        engine.set_probe([&](const std::string& name, int layer, const float* data, size_t n) {
            stages[name + (layer >= 0 ? std::to_string(layer) : "")].assign(data, data + n);
        });
        const auto img = engine.decode(transposed(read_f32(ref / "final_latents.f32"), 128, N), g, g);
        // The decoder's stages ([C][H][W] in the reference, channels last here).
        for (const char* st : {"vae_in", "vae_pq", "vae_conv_in", "vae_mid", "vae_up0", "vae_up1", "vae_up2", "vae_up3"}) {
            const auto p = ref / std::format("{}.f32", st);
            if (!fs::exists(p) || !stages.contains(st))
                continue;
            const auto s = shape(st);
            const auto w = read_f32(p);
            const auto& gs = stages[st];
            if (gs.size() != w.size()) {
                std::println("        {}: {} values, the reference {}", st, gs.size(), w.size());
                continue;
            }
            const Diff ds = diff(transposed(gs, s[1] * s[2], s[0]), w, w.size());
            std::println("        {}: rel error {:.2e}", st, ds.rel);
            if (std::string(st) == "vae_in")
                check(ds.rel < 1e-5, std::format("denormalized latents: rel error {:.2e}", ds.rel));
        }
        std::vector<float> rgb;
        for (size_t p = 0; p < size_t(size) * size; ++p)
            rgb.insert(rgb.end(), img.begin() + p * 4, img.begin() + p * 4 + 3);
        const Diff d = diff(transposed(rgb, size_t(size) * size, 3), read_f32(ref / "vae_out.f32"), size_t(size) * size);
        check(d.rel < 0.02, std::format("{}x{} RGB: rel error {:.2e}, lowest channel cosine {:.5f}", size, size, d.rel,
                                        d.min_cos));
    } else if (mode == "encode") {
        std::map<std::string, std::vector<float>> stages;
        engine.set_probe([&](const std::string& name, int layer, const float* data, size_t n) {
            stages[name + (layer >= 0 ? std::to_string(layer) : "")].assign(data, data + n);
        });
        const auto in = read_f32(ref / "enc_in.f32"); // [3][H][W]
        const size_t px = size_t(size) * size;
        std::vector<float> rgba(px * 4, 1.0f);
        for (size_t p = 0; p < px; ++p)
            for (int c = 0; c < 3; ++c)
                rgba[p * 4 + c] = in[c * px + p];
        const auto lat = engine.encode(rgba, g, g);
        for (const char* st : {"enc_down0", "enc_down1", "enc_down2", "enc_down3", "enc_mid"}) {
            const auto s = shape(st);
            const auto w = read_f32(ref / std::format("{}.f32", st));
            const auto& gs = stages[st];
            if (gs.size() != w.size()) {
                check(false, std::format("{}: {} values, the reference {}", st, gs.size(), w.size()));
                continue;
            }
            const Diff ds = diff(transposed(gs, s[1] * s[2], s[0]), w, w.size());
            std::println("        {}: rel error {:.2e}", st, ds.rel);
        }
        const Diff d = diff(transposed(lat, N, 128), read_f32(ref / "enc_latents.f32"), N);
        check(d.rel < 0.01, std::format("{}x{} latents: rel error {:.2e}, lowest channel cosine {:.5f}", size, size,
                                        d.rel, d.min_cos));
    } else if (mode == "pipeline") {
        const auto sig = engine.sigmas(uint32_t(manifest["steps"].get_number()), N, std::nullopt);
        double worst = 0;
        const auto& ws = manifest["sigmas"].get_array();
        for (size_t i = 0; i < ws.size() && i < sig.size(); ++i)
            worst = std::max(worst, std::abs(double(sig[i]) - ws[i].get_number()));
        check(sig.size() == ws.size() && worst < 1e-6,
              std::format("{} noise levels (the reference's {}), largest difference {:.1e}", sig.size(), ws.size(), worst));
        if (manifest.contains("schedules"))
            for (const auto& [n, want_s] : manifest["schedules"].get_object()) {
                const auto got_s = engine.sigmas(uint32_t(std::stoul(n)), N, std::nullopt);
                const auto& w = want_s.get_array();
                double d = 0;
                for (size_t i = 0; i < w.size() && i < got_s.size(); ++i)
                    d = std::max(d, std::abs(double(got_s[i]) - w[i].get_number()));
                check(got_s.size() == w.size() && d < 1e-6,
                      std::format("{} steps: {} noise levels (the reference's {}), largest difference {:.1e}", n,
                                  got_s.size(), w.size(), d));
            }
        mako::image::Engine::Request r;
        r.prompt = prompt;
        r.width = r.height = size;
        r.steps = uint32_t(manifest["steps"].get_number());
        const auto noise = transposed(read_f32(ref / "c0_latents.f32"), 128, N);
        r.noise = noise;
        const auto img = engine.generate(r);
        std::vector<float> rgb;
        for (size_t p = 0; p < size_t(size) * size; ++p)
            rgb.insert(rgb.end(), img.begin() + p * 4, img.begin() + p * 4 + 3);
        const Diff d = diff(transposed(rgb, size_t(size) * size, 3), read_f32(ref / "vae_out.f32"), size_t(size) * size);
        // Measured: 1.0e-2.
        check(d.rel < 0.03, std::format("{}x{}, {} steps: rel error {:.2e}, lowest channel cosine {:.5f}", size, size,
                                        r.steps, d.rel, d.min_cos));
    } else {
        std::println(stderr, "unknown mode {}", mode);
        return 2;
    }
    if (failures)
        std::println("{} failed", failures);
    return failures ? 1 : 0;
}
