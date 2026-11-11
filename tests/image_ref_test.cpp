// Qwen-Image 2.1 against the reference pipeline's tensors (fp32 on the CPU, 512x512, 3 steps, seed 42; written by a
// script beside the model, manifest.json listing them):
//   tokens   the prompt template's token ids
//   text     the text encoder's output (the transformer's prompt embeddings)
//   dit      the transformer on the reference's prompt embeddings and latents: each stage of the first two steps (the
//            whole sequence, then the image's rows against the cached text) and every step's prediction
//   vae      the VAE decoder on the reference's final latents
//   pipeline the prompt to the image (the reference's starting noise): token ids, noise levels, the image
//   image_ref_test <model dir> <reference dir> <mode>
#include "image/qwen_image.hpp"
#include "mtl/context.hpp"
#include "text/tokenizer.hpp"

#include <glaze/glaze.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <map>
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

} // namespace

int main(int argc, char** argv)
{
    if (argc < 4 || !fs::exists(fs::path(argv[1]) / "transformer") || !fs::exists(fs::path(argv[2]) / "manifest.json")) {
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
    auto tok = mako::text::Tokenizer::load_json((model / "processor" / "tokenizer.json").string());
    if (!tok) {
        std::println(stderr, "{}", tok.error());
        return 1;
    }
    const std::string prompt = manifest["prompt"].get_string();
    std::vector<int32_t> want;
    for (const auto& v : manifest["ids"].get_array())
        want.push_back(int32_t(v.get_number()));

    if (mode == "tokens") {
        const auto ids = tok->encode(mako::image::QwenImage::templated(prompt), true);
        check(ids == want, std::format("the template's {} token ids (the reference's {})", ids.size(), want.size()));
        check(tok->decode(ids) == mako::image::QwenImage::templated(prompt), "they decode to the template");
    } else if (mode == "text") {
        mako::mtl::Context ctx;
        mako::image::QwenImage engine(model.string(), ctx);
        const auto t0 = std::chrono::steady_clock::now();
        const auto got = engine.encode_text(want, uint32_t(manifest["drop_idx"].get_number()));
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        const auto ref_e = read_f32(ref / "prompt_embeds.f32");
        if (got.size() != ref_e.size()) {
            check(false, std::format("{} values, the reference {}", got.size(), ref_e.size()));
        } else {
            // The reference pipeline in BF16 (its usual precision) is 7.8-8.6e-2 off its fp32 output (lowest row
            // cosine 0.91-0.92); mako keeps the residual stream in fp32 (measured 1.2e-2, 0.998).
            const Diff d = diff(got, ref_e, 4096);
            check(d.rel < 0.03 && d.min_cos > 0.995,
                  std::format("{} rows: rel error {:.2e}, lowest row cosine {:.6f} ({:.0f} ms)", got.size() / 4096,
                              d.rel, d.min_cos, ms));
        }
    } else if (mode == "dit") {
        mako::mtl::Context ctx;
        mako::image::QwenImage engine(model.string(), ctx);
        const uint32_t size = uint32_t(manifest["size"].get_number()), g = size / 16, N = g * g;
        std::map<std::string, std::vector<float>> got;
        uint32_t step = 0;
        engine.set_probe([&](const std::string& name, int layer, const float* data, size_t n) {
            if (step < 2)
                got[std::format("s{}_{}{}", step, name, layer >= 0 ? std::to_string(layer) : "")].assign(data, data + n);
        });
        engine.begin(read_f32(ref / "prompt_embeds.f32"), g, g);
        const auto& ts = manifest["timesteps"].get_array();
        for (step = 0; step < ts.size(); ++step) {
            const auto t0 = std::chrono::steady_clock::now();
            const auto pred = engine.step(read_f32(ref / std::format("step{}_latents.f32", step)), float(ts[step].get_number()));
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            auto want_p = read_f32(ref / std::format("step{}_noise_pred.f32", step));
            want_p.erase(want_p.begin(), want_p.end() - ptrdiff_t(N) * 64); // the image's rows
            const Diff d = diff(pred, want_p, 64);
            check(d.rel < 0.05, std::format("step {} (sigma {:.3f}) prediction: rel error {:.2e}, lowest row cosine "
                                            "{:.4f} ({:.0f} ms)",
                                            step, ts[step].get_number(), d.rel, d.min_cos, ms));
        }
        for (const auto& [name, width] : std::vector<std::pair<std::string, size_t>>{
                 {"s0_temb", 4096}, {"s0_modulation", 16384}, {"s0_txt_in", 4096}, {"s0_img_in", 4096},
                 {"s0_block0", 4096}, {"s0_block1", 4096}, {"s0_block15", 4096}, {"s0_block31", 4096},
                 {"s0_norm_out", 4096}, {"s1_img_in", 4096}, {"s1_block0", 4096}, {"s1_block31", 4096},
                 {"s1_norm_out", 4096}}) {
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
    } else if (mode == "vae") {
        mako::mtl::Context ctx;
        mako::image::QwenImage engine(model.string(), ctx);
        const uint32_t size = uint32_t(manifest["size"].get_number()), g = size / 16;
        std::vector<float> vae_in;
        engine.set_probe([&](const std::string& name, int layer, const float* data, size_t n) {
            if (name == "vae_in")
                vae_in.assign(data, data + n);
            if (const char* dump = std::getenv("MAKO_DUMP_VAE"))
                std::ofstream(std::format("{}.{}{}", dump, name, layer >= 0 ? std::to_string(layer) : ""), std::ios::binary)
                    .write(reinterpret_cast<const char*>(data), std::streamsize(n * 4));
        });
        const auto t0 = std::chrono::steady_clock::now();
        const auto img = engine.decode(read_f32(ref / "final_latents.f32"), g, g);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        // The reference's are channels first.
        auto planar = [](const std::vector<float>& v, size_t pixels, size_t c) {
            std::vector<float> o(v.size());
            for (size_t p = 0; p < pixels; ++p)
                for (size_t ch = 0; ch < c; ++ch)
                    o[ch * pixels + p] = v[p * c + ch];
            return o;
        };
        const auto want_in = read_f32(ref / "vae_in.f32");
        const Diff din = diff(planar(vae_in, size_t(g) * g, 64), want_in, size_t(g) * g);
        check(din.rel < 1e-5, std::format("denormalized latents: rel error {:.2e}", din.rel));
        const auto want = read_f32(ref / "vae_out.f32");
        if (img.size() != want.size()) {
            check(false, std::format("{} values, the reference {}", img.size(), want.size()));
        } else {
            const Diff d = diff(planar(img, size_t(size) * size, 4), want, size_t(size) * size);
            check(d.rel < 0.02, std::format("{}x{} RGBA: rel error {:.2e}, lowest channel cosine {:.5f} ({:.0f} ms)",
                                            size, size, d.rel, d.min_cos, ms));
        }
    } else if (mode == "pipeline") {
        mako::mtl::Context ctx;
        mako::image::QwenImage engine(model.string(), ctx);
        const uint32_t size = uint32_t(manifest["size"].get_number()), g = size / 16;
        check(engine.system_tokens() == uint32_t(manifest["drop_idx"].get_number()),
              std::format("the template's system part: {} tokens", engine.system_tokens()));
        const auto sig = engine.sigmas(uint32_t(manifest["steps"].get_number()), g * g);
        double worst = 0;
        const auto& ws = manifest["sigmas"].get_array();
        for (size_t i = 0; i < ws.size() && i < sig.size(); ++i)
            worst = std::max(worst, std::abs(double(sig[i]) - ws[i].get_number()));
        check(sig.size() == ws.size() && worst < 1e-6, std::format("{} noise levels, largest difference {:.1e}", sig.size(), worst));
        const auto noise = read_f32(ref / "step0_latents.f32");
        mako::image::Engine::Request r;
        r.prompt = prompt;
        r.width = r.height = size;
        r.steps = uint32_t(manifest["steps"].get_number());
        r.noise = noise;
        const auto t0 = std::chrono::steady_clock::now();
        const auto img = engine.generate(r);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::vector<float> planar(img.size());
        const size_t px = size_t(size) * size;
        for (size_t p = 0; p < px; ++p)
            for (size_t ch = 0; ch < 4; ++ch)
                planar[ch * px + p] = img[p * 4 + ch];
        const Diff d = diff(planar, read_f32(ref / "vae_out.f32"), px);
        check(d.rel < 0.05, std::format("{}x{}, {} steps: rel error {:.2e}, lowest channel cosine {:.5f} ({:.0f} ms)",
                                        size, size, r.steps, d.rel, d.min_cos, ms));
    } else if (mode == "edit") {
        // The edit path against tools/dump_qwen_image_edit_ref.py: the reference picture as the pipeline resized it
        // goes in, so the encoders see its pixels exactly; the VAE encoder, the text encoder with the picture (its
        // pad rows dropped from the reference's), then the transformer on the reference's text and latents.
        mako::mtl::Context ctx;
        mako::image::QwenImage engine(model.string(), ctx);
        const auto& m = manifest;
        const uint32_t size = uint32_t(m["size"].get_number()), g = size / 16, N = g * g;
        const auto& rs = m["ed_ref0_source_size"].get_array();
        const auto& es = m["ed_ref0_size"].get_array();
        const uint32_t ew = uint32_t(es[0].get_number()), eh = uint32_t(es[1].get_number());
        const auto [gw, gh] = mako::image::QwenImage::reference_size(uint32_t(rs[0].get_number()),
                                                                     uint32_t(rs[1].get_number()));
        check(gw == ew && gh == eh, std::format("the reference's size: {}x{} (the pipeline's {}x{})", gw, gh, ew, eh));
        const std::string u8 = read_file(ref / "ed_ref0.u8");
        std::vector<float> rgba(u8.size());
        for (size_t i = 0; i < u8.size(); ++i)
            rgba[i] = float(uint8_t(u8[i])) / 127.5f - 1.0f;
        mako::image::Engine::Request r;
        r.prompt = m["ed_prompt"].get_string();
        r.width = r.height = size;
        r.references.push_back({ew, eh, rgba});
        engine.set_references(r);
        {
            const auto want_z = read_f32(ref / "ed_ref0_latents.f32");
            const auto got_z = engine.reference_latents(0);
            if (got_z.size() != want_z.size()) {
                check(false, std::format("reference latents: {} values, the reference {}", got_z.size(), want_z.size()));
            } else {
                const Diff d = diff(std::vector<float>(got_z.begin(), got_z.end()), want_z, 64);
                check(d.rel < 2e-2, std::format("reference latents (the VAE encoder): rel error {:.2e}, lowest row cosine {:.5f}", d.rel, d.min_cos));
            }
        }
        const auto full = read_f32(ref / "ed_text_embeds.f32");
        const auto mask = read_f32(ref / "ed_pad_mask.f32");
        std::vector<float> want_text;
        for (size_t row = 0; row < mask.size(); ++row)
            if (mask[row] == 0.0f)
                want_text.insert(want_text.end(), full.begin() + ptrdiff_t(row) * 4096, full.begin() + ptrdiff_t(row + 1) * 4096);
        {
            const auto got_text = engine.prompt_embeddings(r.prompt);
            // The presentation's tokens against the pipeline's.
            std::vector<int32_t> want_ids;
            for (const auto& v : m["ed_ids"].get_array())
                want_ids.push_back(int32_t(v.get_number()));
            const auto& got_ids = engine.presentation();
            size_t first_diff = 0;
            while (first_diff < got_ids.size() && first_diff < want_ids.size() && got_ids[first_diff] == want_ids[first_diff])
                ++first_diff;
            check(got_ids == want_ids, std::format("the presentation: {} tokens (the pipeline's {}), the same up to token {}",
                                                   got_ids.size(), want_ids.size(), first_diff));
            // The vision tower's features and the DeepStack ones.
            const auto& f = engine.reference_features(0);
            {
                const auto want_v = read_f32(ref / "ed_vis_merger.f32");
                if (f.embeds.size() != want_v.size()) {
                    check(false, std::format("vision features: {} values, the reference {}", f.embeds.size(), want_v.size()));
                } else {
                    const Diff d = diff(f.embeds, want_v, f.embeds.size() / (size_t(f.rows) * f.cols));
                    check(d.rel < 5e-2, std::format("vision features: rel error {:.2e}, lowest row cosine {:.5f}", d.rel, d.min_cos));
                }
                for (size_t j = 0; j < f.deepstack.size(); ++j) {
                    const auto want_d = read_f32(ref / std::format("ed_vis_deepstack{}.f32", j));
                    if (f.deepstack[j].size() != want_d.size()) {
                        check(false, std::format("deepstack {}: {} values, the reference {}", j, f.deepstack[j].size(), want_d.size()));
                        continue;
                    }
                    const Diff d = diff(f.deepstack[j], want_d, want_d.size() / (size_t(f.rows) * f.cols));
                    check(d.rel < 5e-2, std::format("deepstack {}: rel error {:.2e}, lowest row cosine {:.5f}", j, d.rel, d.min_cos));
                }
            }
            if (got_text.size() != want_text.size()) {
                check(false, std::format("text embeddings: {} rows, the reference {}", got_text.size() / 4096, want_text.size() / 4096));
            } else {
                const Diff d = diff(got_text, want_text, 4096);
                check(d.rel < 5e-2, std::format("text embeddings with the picture: rel error {:.2e}, lowest row cosine {:.5f}", d.rel, d.min_cos));
                // The rows before the picture (which see no picture) and after it.
                const size_t slot = size_t(m["ed_slots"].get_array()[0].get_number()) * 4096;
                const Diff before = diff(std::vector<float>(got_text.begin(), got_text.begin() + ptrdiff_t(slot)),
                                         std::vector<float>(want_text.begin(), want_text.begin() + ptrdiff_t(slot)), 4096);
                const Diff after = diff(std::vector<float>(got_text.begin() + ptrdiff_t(slot), got_text.end()),
                                        std::vector<float>(want_text.begin() + ptrdiff_t(slot), want_text.end()), 4096);
                std::println("       before the picture: rel error {:.2e}, lowest row cosine {:.5f}; after: rel error {:.2e}, lowest row cosine {:.5f}",
                             before.rel, before.min_cos, after.rel, after.min_cos);
                // Row by row before the picture, with the picture's tokens as plain pad tokens too (those rows see
                // no picture either way).
                const auto plain = engine.encode_text(engine.presentation(), uint32_t(m["ed_drop_idx"].get_number()));
                for (size_t row = 0; row < slot / 4096; ++row) {
                    auto cosine = [&](const std::vector<float>& a, size_t ar, const std::vector<float>& b, size_t br) {
                        double ab = 0, aa = 0, bb = 0;
                        for (size_t i = 0; i < 4096; ++i) {
                            ab += double(a[ar * 4096 + i]) * b[br * 4096 + i];
                            aa += double(a[ar * 4096 + i]) * a[ar * 4096 + i];
                            bb += double(b[br * 4096 + i]) * b[br * 4096 + i];
                        }
                        return ab / std::sqrt(aa * bb);
                    };
                    std::println("       row {}: with the picture {:.4f}, as plain tokens {:.4f}", row, cosine(got_text, row, want_text, row),
                                 cosine(plain, row, want_text, row));
                }
            }
        }
        engine.set_reference_latents(0, read_f32(ref / "ed_ref0_latents.f32"));
        const uint32_t steps = uint32_t(m["ed_steps"].get_number());
        const auto sig = engine.sigmas(steps, N);
        const auto& ws = m["ed_sigmas"].get_array();
        double worst = 0;
        for (size_t i = 0; i < ws.size() && i < sig.size(); ++i)
            worst = std::max(worst, std::abs(double(sig[i]) - ws[i].get_number()));
        check(sig.size() == ws.size() && worst < 1e-5, std::format("{} noise levels, largest difference {:.1e}", sig.size(), worst));
        std::map<std::string, std::vector<float>> got;
        uint32_t step = 0;
        engine.set_probe([&](const std::string& name, int layer, const float* data, size_t n) {
            got[std::format("s{}_{}{}", step, name, layer >= 0 ? std::to_string(layer) : "")].assign(data, data + n);
        });
        engine.begin(want_text, g, g);
        for (step = 0; step < steps; ++step) {
            const auto t0 = std::chrono::steady_clock::now();
            const auto pred = engine.step(read_f32(ref / std::format("ed_step{}_latents.f32", step)), float(ws[step].get_number()));
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            const auto want_p = read_f32(ref / std::format("ed_step{}_pred.f32", step));
            const Diff d = diff(pred, want_p, 64);
            check(d.rel < 0.05, std::format("step {} (sigma {:.3f}) prediction: rel error {:.2e}, lowest row cosine {:.4f} ({:.0f} ms)",
                                            step, ws[step].get_number(), d.rel, d.min_cos, ms));
            // The joint sequence's rows the engine ran this step (all of them at the first, the image's after).
            for (const char* name : {"block0", "block1", "block15", "block31", "norm_out"}) {
                const auto it = got.find(std::format("s{}_{}", step, name));
                auto want_s = read_f32(ref / std::format("ed_s{}_{}.f32", step, name));
                if (it == got.end() || it->second.size() > want_s.size() || it->second.size() % 4096) {
                    check(false, std::format("{}: {} values, the reference {}", name, it == got.end() ? 0 : it->second.size(), want_s.size()));
                    continue;
                }
                want_s.erase(want_s.begin(), want_s.end() - ptrdiff_t(it->second.size()));
                const Diff ds = diff(it->second, want_s, 4096);
                std::println("  {:4} s{}_{}: rel error {:.2e}, lowest row cosine {:.5f}", "", step, name, ds.rel, ds.min_cos);
            }
        }
    } else {
        std::println(stderr, "unknown mode {}", mode);
        return 2;
    }
    if (failures)
        std::println("{} failed", failures);
    return failures ? 1 : 0;
}
