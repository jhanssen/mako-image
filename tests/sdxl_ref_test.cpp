// SDXL against the reference implementation's tensors (fp32 on the CPU, 256x256, 3 Euler steps with its normal
// schedule, guidance 6, seed 42; written by a script beside the checkpoint, manifest.json listing them):
//   tokens   the tokenizer's ids for odd texts, and the prompt's chunks with their weights
//   text     each encoder's states per chunk, the weighted conditioning and the pooled state
//   unet     the UNet on the reference's conditioning and inputs: stages of the first pass and every pass's noise
//   vae      the VAE decoder on the reference's final latents
//   pipeline the prompt to the image (the reference's starting noise)
//   lora     a LoRA merged into both text encoders and the UNet (its file the fourth argument; the reference's
//            lora_manifest.json): the prompt's conditioning and the first prompt pass's noise
//   sdxl_ref_test <model dir> <reference dir> <mode> [LoRA file]
#include "image/sdxl.hpp"
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

struct Diff {
    double rel = 0, min_cos = 1;
};
Diff diff(const std::vector<float>& got, const std::vector<float>& want, size_t width)
{
    Diff d;
    double e = 0, r = 0;
    for (size_t i = 0; i < want.size() && i < got.size(); ++i) {
        e += (double(got[i]) - want[i]) * (double(got[i]) - want[i]);
        r += double(want[i]) * want[i];
    }
    d.rel = std::sqrt(e / r);
    for (size_t row = 0; row < want.size() / width && (row + 1) * width <= got.size(); ++row) {
        double gw = 0, gg = 0, ww = 0;
        for (size_t i = row * width; i < (row + 1) * width; ++i) {
            gw += double(got[i]) * want[i];
            gg += double(got[i]) * got[i];
            ww += double(want[i]) * want[i];
        }
        if (ww > 0)
            d.min_cos = std::min(d.min_cos, gw / std::sqrt(gg * ww));
    }
    return d;
}

// Latents [4][2g][2g] to tokens [g * g][16] (a token's channel, row, column) and back.
std::vector<float> to_tokens(const std::vector<float>& v, uint32_t g, float scale = 1.0f)
{
    std::vector<float> o(v.size());
    for (uint32_t c = 0; c < 4; ++c)
        for (uint32_t y = 0; y < 2 * g; ++y)
            for (uint32_t x = 0; x < 2 * g; ++x)
                o[((y / 2) * g + x / 2) * 16 + c * 4 + (y % 2) * 2 + x % 2] =
                    v[(uint64_t(c) * 2 * g + y) * 2 * g + x] * scale;
    return o;
}
std::vector<float> from_tokens(const std::vector<float>& t, uint32_t g)
{
    std::vector<float> o(t.size());
    for (uint32_t c = 0; c < 4; ++c)
        for (uint32_t y = 0; y < 2 * g; ++y)
            for (uint32_t x = 0; x < 2 * g; ++x)
                o[(uint64_t(c) * 2 * g + y) * 2 * g + x] = t[((y / 2) * g + x / 2) * 16 + c * 4 + (y % 2) * 2 + x % 2];
    return o;
}

// [C][H * W] to channels last.
std::vector<float> channels_last(const std::vector<float>& v, size_t c)
{
    const size_t px = v.size() / c;
    std::vector<float> o(v.size());
    for (size_t p = 0; p < px; ++p)
        for (size_t k = 0; k < c; ++k)
            o[p * c + k] = v[k * px + p];
    return o;
}

std::vector<int32_t> ids(const glz::generic& v)
{
    std::vector<int32_t> out;
    for (const auto& x : v.get_array())
        out.push_back(int32_t(x.get_number()));
    return out;
}

std::vector<int32_t> column(const std::vector<mako::image::Sdxl::Token>& c)
{
    std::vector<int32_t> out;
    for (const auto& t : c)
        out.push_back(t.id);
    return out;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 4 || mako::image::Sdxl::checkpoint(argv[1]).empty() || !fs::exists(fs::path(argv[2]) / "manifest.json")) {
        std::println("model {} or reference {} not found: skipped", argc > 1 ? argv[1] : "", argc > 2 ? argv[2] : "");
        return 77;
    }
    const fs::path ref = argv[2];
    const std::string mode = argv[3];
    glz::generic manifest;
    if (glz::read_json(manifest, read_file(ref / "manifest.json"))) {
        std::println(stderr, "bad manifest");
        return 1;
    }
    std::println("{}", mode);
    const std::string prompt = manifest["prompt"].get_string(), negative = manifest["negative"].get_string();
    const uint32_t size = uint32_t(manifest["size"].get_number()), g = size / 16;
    if (mode == "tokens") {
        auto tok = mako::image::ClipTokenizer::load(fs::path(argv[1]) / "tokenizer" / "vocab.json",
                                                    fs::path(argv[1]) / "tokenizer" / "merges.txt");
        if (!tok) {
            check(false, tok.error());
            return 1;
        }
        for (const auto& c : manifest["tok_cases"].get_array()) {
            const auto got = tok->encode(c["text"].get_string());
            const auto want = ids(c["ids"]);
            check(got == want, std::format("{:?}: {} tokens (the reference's {})", c["text"].get_string(), got.size(),
                                           want.size()));
        }
    }
    mako::mtl::Context ctx;
    mako::image::Sdxl engine(argv[1], ctx);
    // The conditioning as prompt_embeddings packs it, from the reference's tensors.
    auto packed = [&](const std::string& name) {
        const auto cond = read_f32(ref / (name + "_cond.f32"));
        const auto pooled = read_f32(ref / (name + "_pooled.f32"));
        std::vector<float> p{float(cond.size() / 2048)};
        p.insert(p.end(), cond.begin(), cond.end());
        p.insert(p.end(), pooled.begin(), pooled.end());
        return p;
    };

    if (mode == "tokens") {
        auto chunks_match = [&](const std::string& text, const glz::generic& l, const glz::generic& gg,
                                const glz::generic& w) {
            const auto cl = engine.chunks(text, mako::image::Sdxl::kPadL);
            const auto cg = engine.chunks(text, mako::image::Sdxl::kPadG);
            bool ok = cl.size() == l.get_array().size() && cg.size() == cl.size();
            for (size_t k = 0; ok && k < cl.size(); ++k) {
                ok = column(cl[k]) == ids(l.get_array()[k]) && column(cg[k]) == ids(gg.get_array()[k]);
                const auto& wk = w.get_array()[k].get_array();
                for (size_t t = 0; ok && t < cl[k].size(); ++t)
                    ok = std::abs(cl[k][t].weight - float(wk[t].get_number())) < 1e-6f;
            }
            check(ok, std::format("{:.40?}: {} chunks (the reference's {})", text, cl.size(), l.get_array().size()));
        };
        chunks_match(prompt, manifest["pos_l_ids"], manifest["pos_g_ids"], manifest["pos_weights"]);
        chunks_match(negative, manifest["neg_l_ids"], manifest["neg_g_ids"], manifest["neg_weights"]);
        for (const auto& c : manifest["chunk_cases"].get_array())
            chunks_match(c["text"].get_string(), c["l"], c["g"], c["w"]);
    } else if (mode == "text") {
        for (const char* name : {"pos", "neg"}) {
            const auto wl = read_f32(ref / std::format("{}_l.f32", name));
            const auto wg = read_f32(ref / std::format("{}_g.f32", name));
            const auto& lc = manifest[std::string(name) + "_l_ids"].get_array();
            const auto& gc = manifest[std::string(name) + "_g_ids"].get_array();
            for (size_t k = 0; k < lc.size(); ++k) {
                const auto zl = engine.encode_clip(0, ids(lc[k]));
                const auto zg = engine.encode_clip(1, ids(gc[k]));
                const std::vector<float> rl(wl.begin() + ptrdiff_t(k * 77 * 768), wl.begin() + ptrdiff_t((k + 1) * 77 * 768));
                const std::vector<float> rg(wg.begin() + ptrdiff_t(k * 77 * 1280),
                                            wg.begin() + ptrdiff_t((k + 1) * 77 * 1280));
                const Diff dl = diff(zl, rl, 768), dg = diff(zg, rg, 1280);
                check(dl.rel < 0.01, std::format("{} chunk {} ViT-L: rel error {:.2e}, lowest row cosine {:.5f}", name, k,
                                                 dl.rel, dl.min_cos));
                check(dg.rel < 0.01, std::format("{} chunk {} bigG: rel error {:.2e}, lowest row cosine {:.5f}", name, k,
                                                 dg.rel, dg.min_cos));
            }
            const auto p = engine.prompt_embeddings(name == std::string("pos") ? prompt : negative);
            const auto want = packed(name);
            const uint32_t S = uint32_t(p[0]);
            const std::vector<float> cond(p.begin() + 1, p.begin() + 1 + ptrdiff_t(S) * 2048);
            const std::vector<float> wcond(want.begin() + 1, want.end() - 1280);
            const Diff dc = diff(cond, wcond, 2048);
            check(S == uint32_t(want[0]) && dc.rel < 0.01,
                  std::format("{}: {} rows weighted: rel error {:.2e}, lowest row cosine {:.5f}", name, S, dc.rel,
                              dc.min_cos));
            const Diff dp = diff(std::vector<float>(p.end() - 1280, p.end()), std::vector<float>(want.end() - 1280, want.end()),
                                 1280);
            check(dp.rel < 0.01, std::format("{}: pooled: rel error {:.2e}", name, dp.rel));
        }
    } else if (mode == "unet") {
        const auto& sig = manifest["sigmas"].get_array();
        const auto& passes = manifest["passes"].get_array();
        // Stages of the first pass.
        std::map<std::string, std::vector<float>> got;
        bool first = true;
        engine.set_probe([&](const std::string& name, int layer, const float* data, size_t n) {
            if (first)
                got[layer >= 0 ? std::format("{}{}", name, layer) : name].assign(data, data + n);
        });
        engine.begin(packed("pos"), g, g, 0);
        engine.begin(packed("neg"), g, g, 1);
        for (size_t k = 0; k < passes.size(); ++k) {
            const auto& p = passes[k];
            const float s = float(sig[k / 2].get_number());
            const uint32_t slot = p["prompt"].get_string() == "pos" ? 0 : 1;
            check(engine.timestep(s) == uint32_t(p["t"].get_number()),
                  std::format("pass {}: timestep {} (the reference's {})", k, engine.timestep(s), p["t"].get_number()));
            const auto x = to_tokens(read_f32(ref / std::format("p{}_x.f32", k)), g, std::sqrt(s * s + 1.0f));
            const auto eps = from_tokens(engine.step(x, s, slot), g);
            first = false;
            const Diff d = diff(eps, read_f32(ref / std::format("p{}_out.f32", k)), size_t(4 * g * g));
            check(d.rel < 0.02, std::format("pass {} ({}, sigma {:.3f}): rel error {:.2e}, lowest channel cosine {:.5f}",
                                            k, p["prompt"].get_string(), s, d.rel, d.min_cos));
        }
        std::println("  stages of pass 0 ({}):", manifest["stage_prompt"].get_string());
        const auto& tensors = manifest["tensors"].get_object();
        for (const auto& [name, shape] : tensors) {
            if (!name.starts_with("s_"))
                continue;
            const std::string key = name.substr(2);
            const auto it = got.find(key);
            auto want = read_f32(ref / (name + ".f32"));
            const size_t c = size_t(shape.get_array()[0].get_number());
            if (shape.get_array().size() == 3)
                want = channels_last(want, c);
            if (it == got.end() || it->second.size() != want.size()) {
                check(false, std::format("{}: {} values, the reference {}", key, it == got.end() ? 0 : it->second.size(),
                                         want.size()));
                continue;
            }
            const Diff d = diff(it->second, want, shape.get_array().size() == 3 ? c : want.size());
            std::println("        {}: rel error {:.2e}, lowest row cosine {:.5f}", key, d.rel, d.min_cos);
        }
    } else if (mode == "vae" || mode == "pipeline") {
        std::vector<float> img;
        if (mode == "vae") {
            img = engine.decode(to_tokens(read_f32(ref / "final_latents.f32"), g, 0.13025f), g, g);
        } else {
            const auto sig = engine.sigmas(uint32_t(manifest["steps"].get_number()), g * g, std::nullopt);
            const auto& ws = manifest["sigmas"].get_array();
            double worst = 0;
            for (size_t i = 0; i < ws.size() && i < sig.size(); ++i)
                worst = std::max(worst, std::abs(double(sig[i]) - ws[i].get_number()) / std::max(1e-3, ws[i].get_number()));
            check(sig.size() == ws.size() && worst < 1e-6,
                  std::format("{} noise levels (the reference's {}), largest relative difference {:.1e}", sig.size(),
                              ws.size(), worst));
            mako::image::Engine::Request r;
            r.prompt = prompt;
            r.negative = negative;
            r.guidance = float(manifest["cfg"].get_number());
            r.width = r.height = size;
            r.steps = uint32_t(manifest["steps"].get_number());
            r.sampler = mako::image::Sampler::euler;
            const auto noise = to_tokens(read_f32(ref / "noise.f32"), g);
            r.noise = noise;
            std::vector<float> vae_in;
            engine.set_probe([&](const std::string& name, int, const float* data, size_t n) {
                if (name == "vae_in")
                    vae_in.assign(data, data + n);
            });
            img = engine.generate(r);
            // The final latents, channels last there.
            std::println("        final latents: rel error {:.2e}",
                         diff(channels_last(read_f32(ref / "final_latents.f32"), 4), vae_in, 4).rel);
        }
        std::vector<float> rgb;
        for (size_t p = 0; p < size_t(size) * size; ++p)
            for (int c = 0; c < 3; ++c)
                rgb.push_back((img[p * 4 + c] + 1.0f) / 2.0f);
        const Diff d = diff(rgb, read_f32(ref / "image.f32"), 3);
        check(d.rel < (mode == "vae" ? 0.01 : 0.05),
              std::format("{}x{} RGB: rel error {:.2e}, lowest pixel cosine {:.4f}", size, size, d.rel, d.min_cos));
    } else if (mode == "lora") {
        glz::generic lm;
        if (argc < 5 || !fs::exists(argv[4]) || glz::read_json(lm, read_file(ref / "lora_manifest.json"))) {
            std::println("no LoRA file or lora_manifest.json: skipped");
            return 77;
        }
        engine.add_lora(argv[4], float(lm["scale"].get_number()));
        const auto p = engine.prompt_embeddings(prompt);
        const auto wc = read_f32(ref / "lora_pos_cond.f32"), wp = read_f32(ref / "lora_pos_pooled.f32");
        const Diff dc = diff(std::vector<float>(p.begin() + 1, p.end() - 1280), wc, 2048);
        check(dc.rel < 0.01, std::format("conditioning: rel error {:.2e} (without the LoRA {:.2e})", dc.rel,
                                         diff(read_f32(ref / "pos_cond.f32"), wc, 2048).rel));
        const Diff dp = diff(std::vector<float>(p.end() - 1280, p.end()), wp, 1280);
        check(dp.rel < 0.01, std::format("pooled: rel error {:.2e} (without the LoRA {:.2e})", dp.rel,
                                         diff(read_f32(ref / "pos_pooled.f32"), wp, 1280).rel));
        // The pass on the reference's conditioning without the LoRA, as the reference ran it.
        const uint32_t k = uint32_t(lm["pass"].get_number());
        const float s = float(manifest["sigmas"].get_array()[k / 2].get_number());
        engine.begin(packed("pos"), g, g, 0);
        const auto x = to_tokens(read_f32(ref / std::format("p{}_x.f32", k)), g, std::sqrt(s * s + 1.0f));
        const auto eps = from_tokens(engine.step(x, s, 0), g);
        const auto want = read_f32(ref / "lora_pass_out.f32");
        const Diff d = diff(eps, want, size_t(4 * g * g));
        check(d.rel < 0.02, std::format("pass {}: rel error {:.2e} (without the LoRA {:.2e})", k, d.rel,
                                        diff(read_f32(ref / std::format("p{}_out.f32", k)), want, 4 * g * g).rel));
    } else if (mode != "tokens") {
        std::println(stderr, "unknown mode {}", mode);
        return 2;
    }
    if (failures)
        std::println("{} failed", failures);
    return failures ? 1 : 0;
}
