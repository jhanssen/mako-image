// MiniMax-H3-Image against the Diffusers implementation's tensors (tools/dump_h3_image_ref.py: the text encoder in
// fp32, the release's transformer with its BF16 weights computed in fp32, the VAE in fp32; 256x256, the checkpoint's
// six steps). The model directory may hold the ComfyUI repack's pruned transformer (--transformer): its curve-table
// modulation and stacked heads are what the transformer checks then measure against the release's full projections.
//   tokens    the prompt's and other strings' token ids
//   schedule  the grid's sigmas and timesteps
//   text      the text encoder's 50th layer output
//   dit       the transformer on the reference's text and rows: every step's stages and velocity
//   pipeline  the engine end to end from the reference's noise, against its decoded image
//   vae       the VAE decoder on the reference's final latents
//   encode    the VAE encoder on the reference's image: the posterior and the sampled, normalized latents
//   edit_text the edit presentation (labels, vision blocks, prompt), the vision tower and the text encoder over it
//   edit      the transformer with two references' rows in the sequence: the first two steps
//   h3_image_ref_test <model dir> <reference dir> <mode> [transformer file]
#include "image/h3_image.hpp"
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
    if (s.empty())
        throw std::runtime_error("missing " + p.string());
    std::vector<float> v(s.size() / 4);
    std::memcpy(v.data(), s.data(), v.size() * 4);
    return v;
}

std::vector<uint8_t> read_u8(const fs::path& p)
{
    const std::string s = read_file(p);
    if (s.empty())
        throw std::runtime_error("missing " + p.string());
    return {s.begin(), s.end()};
}

// Relative RMS error over everything, and the lowest cosine similarity of a row of `width`.
struct Diff {
    double rel = 0, min_cos = 1;
};
Diff diff(std::span<const float> got, std::span<const float> want, size_t width)
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

void compare(const std::string& what, std::span<const float> got, const std::vector<float>& want, size_t width,
             double tol)
{
    if (got.size() != want.size()) {
        check(false, std::format("{}: {} values, the reference {}", what, got.size(), want.size()));
        return;
    }
    const Diff d = diff(got, want, width);
    check(d.rel < tol, std::format("{}: rel error {:.2e}, lowest row cosine {:.5f} (tolerance {:.0e})", what, d.rel,
                                   d.min_cos, tol));
}

std::vector<int32_t> ids_of(const glz::generic& v)
{
    std::vector<int32_t> out;
    for (const auto& x : v.get_array())
        out.push_back(int32_t(x.get_number()));
    return out;
}

std::vector<uint8_t> tags_of(const glz::generic& v)
{
    std::vector<uint8_t> out;
    for (const auto& x : v.get_array())
        out.push_back(uint8_t(x.get_number()));
    return out;
}

std::vector<float> floats_of(const glz::generic& v)
{
    std::vector<float> out;
    for (const auto& x : v.get_array())
        out.push_back(float(x.get_number()));
    return out;
}

// Packed rows [h/2 * w/2][96] (a patch's [c][dy][dx]) as latents [h][w][24], and back.
std::vector<float> unpacked(std::span<const float> rows, uint32_t h, uint32_t w)
{
    std::vector<float> z(uint64_t(h) * w * 24);
    for (uint32_t py = 0; py < h / 2; ++py)
        for (uint32_t px = 0; px < w / 2; ++px)
            for (uint32_t c = 0; c < 24; ++c)
                for (uint32_t dy = 0; dy < 2; ++dy)
                    for (uint32_t dx = 0; dx < 2; ++dx)
                        z[(uint64_t(2 * py + dy) * w + 2 * px + dx) * 24 + c] =
                            rows[(uint64_t(py) * (w / 2) + px) * 96 + (c * 2 + dy) * 2 + dx];
    return z;
}

std::vector<float> packed(std::span<const float> z, uint32_t h, uint32_t w)
{
    std::vector<float> rows(uint64_t(h / 2) * (w / 2) * 96);
    for (uint32_t py = 0; py < h / 2; ++py)
        for (uint32_t px = 0; px < w / 2; ++px)
            for (uint32_t c = 0; c < 24; ++c)
                for (uint32_t dy = 0; dy < 2; ++dy)
                    for (uint32_t dx = 0; dx < 2; ++dx)
                        rows[(uint64_t(py) * (w / 2) + px) * 96 + (c * 2 + dy) * 2 + dx] =
                            z[(uint64_t(2 * py + dy) * w + 2 * px + dx) * 24 + c];
    return rows;
}

// 8-bit RGB [h][w][3] as RGBA floats in [-1, 1].
std::vector<float> rgba_of(std::span<const uint8_t> rgb)
{
    std::vector<float> out(rgb.size() / 3 * 4, 1.0f);
    for (size_t p = 0; p < rgb.size() / 3; ++p)
        for (int c = 0; c < 3; ++c)
            out[p * 4 + c] = float(rgb[p * 3 + c]) / 255.0f * 2.0f - 1.0f;
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
    glz::generic m;
    if (glz::read_json(m, read_file(ref / "manifest.json"))) {
        std::println(stderr, "bad manifest");
        return 1;
    }
    std::println("{}", mode);
    using mako::image::H3Image;
    using mako::models::H3;
    const std::string prompt = m["prompt"].get_string();
    const uint32_t size = uint32_t(m["size"].get_number()), lat = size / 16, N = lat * lat;
    mako::mtl::Context ctx;
    mako::image::Options opts;
    if (argc > 4)
        opts.transformer = argv[4];
    // The repack's fp16 VAE when the directory has it, the release's otherwise.
    if (const fs::path v = model / "comfy/vae/minimax_h3_image_vae_fp16.safetensors"; fs::exists(v))
        opts.vae = v.string();
    H3Image engine(model.string(), ctx, opts);
    const bool pruned = argc > 4;
    // Where the transformer and the text encoder do not fit together, the engine takes turns (the release's 66 GB
    // shards on a 128 GB Mac): the modes that run both pay a reload or two.
    std::println("  the transformer and the text encoder {}", engine.keeps_both() ? "stay loaded together" : "take turns");

    if (mode == "tokens") {
        for (const auto& c : m["token_cases"].get_array()) {
            const std::string text = c["text"].get_string();
            const auto got = engine.tokenizer().encode(text, true);
            const auto want = ids_of(c["ids"]);
            check(got == want, std::format("\"{}\": {} token ids (the reference's {})",
                                           text.size() > 30 ? text.substr(0, 30) + "..." : text, got.size(), want.size()));
        }
    } else if (mode == "schedule") {
        const auto want = floats_of(m["sigmas"]), ts = floats_of(m["timesteps"]);
        const auto got = engine.sigmas(engine.default_steps(), N, std::nullopt);
        bool same = got.size() == want.size();
        for (size_t i = 0; same && i < got.size(); ++i)
            same = got[i] == want[i] && (i + 1 == got.size() || 1.0f - got[i] == ts[i]);
        check(same, std::format("{} sigmas, bit-equal to the reference's grid and timesteps", got.size()));
    } else if (mode == "text") {
        const auto got = engine.prompt_embeddings(prompt);
        compare("text embeddings (layer 50)", got, read_f32(ref / "text_embeds.f32"), H3::kText, 0.03);
    } else if (mode == "dit") {
        const auto text = read_f32(ref / "text_embeds.f32");
        H3& dit = engine.transformer();
        std::map<std::string, std::vector<float>> probes;
        dit.set_probe([&](const std::string& name, int layer, const float* data, size_t n) {
            probes[layer < 0 ? name : std::format("{}{}", name, layer)] = std::vector<float>(data, data + n);
        });
        engine.begin(text, lat, lat);
        compare("text after the refiner", probes["txt"], read_f32(ref / "s0_txt.f32"), H3::kDim, 5e-3);
        const auto sig = floats_of(m["sigmas"]);
        const uint32_t steps = uint32_t(m["steps"].get_number());
        for (uint32_t i = 0; i < steps; ++i) {
            probes.clear();
            const auto rows = read_f32(ref / std::format("step{}_rows.f32", i));
            const auto vel = engine.step(unpacked(rows, lat, lat), sig[i]);
            std::println(" step {} (sigma {})", i, sig[i]);
            // Pruned transformers skip the time embedding; the full one's goes through the MLP the reference has.
            if (probes.contains("temb"))
                compare("time embedding", probes["temb"], read_f32(ref / std::format("s{}_temb.f32", i)), H3::kTimeDim, 1e-4);
            {
                // The reference's modulation: [6][U * 3][5376] chunks; mako's the projection's rows [U][3][6][5376].
                const auto want = read_f32(ref / std::format("s{}_mod0.f32", i));
                const auto& got = probes["mod0"];
                const uint32_t D = H3::kDim, U = uint32_t(want.size() / (18 * D));
                std::vector<float> g(want.size());
                if (got.size() == want.size())
                    for (uint32_t c = 0; c < 6; ++c)
                        for (uint32_t u = 0; u < U; ++u)
                            for (uint32_t md = 0; md < 3; ++md)
                                std::copy_n(got.begin() + (uint64_t(u) * 18 + md * 6 + c) * D, D,
                                            g.begin() + (uint64_t(c) * U * 3 + u * 3 + md) * D);
                compare("block 0 modulation", g, want, D, 2e-2);
            }
            compare("packed input", probes["x0"], read_f32(ref / std::format("s{}_x0.f32", i)), H3::kDim, 5e-3);
            // The pruned transformer's curve-table modulation moves the late blocks more (its velocities not).
            for (int b : {0, 1, 24, 49})
                compare(std::format("block {}", b), probes[std::format("block{}", b)],
                        read_f32(ref / std::format("s{}_block{}.f32", i, b)), H3::kDim, b < 40 ? 2e-2 : pruned ? 8e-2 : 5e-2);
            compare("final norm", probes["norm_out"], read_f32(ref / std::format("s{}_norm_out.f32", i)), H3::kDim,
                    pruned ? 0.15 : 6e-2);
            // The engine's velocity points along the noise; the reference's at the data.
            std::vector<float> want = read_f32(ref / std::format("step{}_pred.f32", i));
            for (float& v : want)
                v = -v;
            compare("velocity", packed(vel, lat, lat), want, 96, 3e-2);
        }
    } else if (mode == "pipeline") {
        mako::image::Engine::Request r;
        r.prompt = prompt;
        r.width = r.height = size;
        r.min_area = 0;
        const auto noise = unpacked(read_f32(ref / "noise_rows.f32"), lat, lat);
        r.noise = noise;
        const auto img = engine.generate(r);
        const auto want = read_u8(ref / "vae_out.u8");
        std::vector<float> got(want.size()), w(want.size());
        for (size_t p = 0; p < want.size() / 3; ++p)
            for (int c = 0; c < 3; ++c) {
                got[p * 3 + c] = std::round((img[p * 4 + c] * 0.5f + 0.5f) * 255.0f);
                w[p * 3 + c] = float(want[p * 3 + c]);
            }
        double sum = 0;
        int worst = 0;
        for (size_t i = 0; i < w.size(); ++i) {
            sum += std::abs(got[i] - w[i]);
            worst = std::max(worst, int(std::abs(got[i] - w[i])));
        }
        const Diff d = diff(got, w, 3);
        check(d.rel < 0.08, std::format("{}x{}, {} steps: rel error {:.2e}, mean |diff| {:.2f} of 255, worst {}", size,
                                        size, engine.default_steps(), d.rel, sum / double(w.size()), worst));
    } else if (mode == "vae") {
        const uint32_t C = uint32_t(m["reference_rows"].get_number());
        const auto rows = read_f32(ref / "final_rows.f32");
        const auto z = unpacked(std::span(rows).subspan(uint64_t(C) * 96), lat, lat);
        const auto rgb = engine.vae().decode(z, 1, lat, lat);
        compare("decoded image", rgb, read_f32(ref / "vae_out.f32"), 3, 0.02);
        const auto img = engine.decode(z, lat, lat);
        const auto want = read_u8(ref / "vae_out.u8");
        size_t off = 0;
        int worst = 0;
        for (size_t p = 0; p < want.size() / 3; ++p)
            for (int c = 0; c < 3; ++c) {
                const int g = int(std::lround((img[p * 4 + c] * 0.5f + 0.5f) * 255.0f));
                off += g != want[p * 3 + c];
                worst = std::max(worst, std::abs(g - want[p * 3 + c]));
            }
        // F16 weights against the reference's fp32: a level or two off in a few percent of the samples.
        check(worst <= 2 && off < want.size() / 10,
              std::format("8-bit image: {} of {} samples differ, worst by {}", off, want.size(), worst));
    } else if (mode == "encode") {
        const auto rgb = read_u8(ref / "enc_in.u8");
        std::vector<float> mean, sd;
        engine.vae().encode_moments(rgb, size, size, mean, sd);
        compare("posterior mean", mean, read_f32(ref / "enc_mean.f32"), 24, 0.01);
        compare("posterior deviation", sd, read_f32(ref / "enc_std.f32"), 24, 0.02);
        const auto z = engine.vae().encode(rgb, size, size, read_f32(ref / "enc_noise.f32"));
        compare("normalized latents", z, read_f32(ref / "enc_latents.f32"), 24, 0.01);
    } else if (mode == "edit_text" || mode == "edit") {
        // The references go in as the pipeline's resized copies (its text-encoder and VAE copies are the same
        // picture here, both at a 1024 short edge), so the engine resizes nothing and the tower sees the reference's
        // pixels exactly: Qwen3-VL's tower moves 30% (PyTorch, fp32) on the rounding differences between two Lanczos
        // implementations over these smooth synthetic pictures. The engine's own resize is checked on its own below.
        mako::image::Engine::Request r;
        r.prompt = m["ed_prompt"].get_string();
        r.width = r.height = size;
        r.min_area = 0;
        r.seed = 3;
        std::vector<std::vector<float>> rgba, noises;
        std::vector<mako::image::Engine::Request::Reference> refs;
        std::vector<std::pair<uint32_t, uint32_t>> vsize, osize;
        for (int i = 0; i < 2; ++i) {
            const auto& vs = m[std::format("ed_vision{}_size", i)].get_array();
            const auto& es = m[std::format("ed_vae{}_size", i)].get_array();
            const auto& os = m[std::format("ed_ref{}_size", i)].get_array();
            vsize.emplace_back(uint32_t(vs[0].get_number()), uint32_t(vs[1].get_number()));
            osize.emplace_back(uint32_t(os[0].get_number()), uint32_t(os[1].get_number()));
            if (uint32_t(es[0].get_number()) != vsize[i].first || uint32_t(es[1].get_number()) != vsize[i].second) {
                std::println(stderr, "reference {}: the dump's VAE copy differs in size from its text-encoder copy", i + 1);
                return 2;
            }
            rgba.push_back(rgba_of(read_u8(ref / std::format("ed_vision{}.u8", i))));
            refs.push_back({vsize[i].first, vsize[i].second, rgba.back()});
            noises.push_back(read_f32(ref / std::format("ed_ref{}_sample_noise.f32", i)));
            noises.push_back(read_f32(ref / std::format("ed_ref{}_mix_noise.f32", i)));
        }
        r.references = refs;
        for (const auto& n : noises)
            r.reference_noise.emplace_back(n);
        if (mode == "edit_text") {
            for (int i = 0; i < 2; ++i) {
                const auto [w, h] = osize[i];
                // The sizes the engine picks for the originals, against the pipeline's.
                const auto [vw, vh] = H3Image::bound_short_edge(w, h, H3Image::kVisionShortEdge, H3Image::kVisionShortEdge);
                const auto [ew, eh] = H3Image::bound_short_edge(w, h, H3Image::kRefMinShortEdge, H3Image::kRefMaxShortEdge);
                check(vw == vsize[i].first && vh == vsize[i].second,
                      std::format("reference {} for the text encoder: {}x{}", i + 1, vw, vh));
                check(ew == vsize[i].first && eh == vsize[i].second, std::format("reference {} for the VAE: {}x{}", i + 1, ew, eh));
                // The engine's resize of the original (Lanczos in [-1, 1], then 8-bit) against the pipeline's (PIL's).
                const auto orig = rgba_of(read_u8(ref / std::format("ed_ref{}.u8", i)));
                const auto sized = mako::image::resize_image(orig, w, h, vw, vh);
                const auto& want = refs[i].rgba;
                double sum = 0;
                int worst = 0;
                uint64_t off = 0;
                for (uint64_t p = 0; p < uint64_t(vw) * vh; ++p)
                    for (uint32_t c = 0; c < 3; ++c) {
                        const int got = int(std::lround(std::clamp(sized[p * 4 + c] * 0.5f + 0.5f, 0.0f, 1.0f) * 255.0f));
                        const int d = std::abs(got - int(std::lround(std::clamp(want[p * 4 + c] * 0.5f + 0.5f, 0.0f, 1.0f) * 255.0f)));
                        sum += d;
                        worst = std::max(worst, d);
                        off += d > 2;
                    }
                check(worst <= 32 && off < uint64_t(vw) * vh * 3 / 1000,
                      std::format("reference {} resized by the engine: mean |diff| {:.3f} levels, worst {}, {} of {} samples "
                                  "more than 2 off",
                                  i + 1, sum / (double(vw) * vh * 3), worst, off, uint64_t(vw) * vh * 3));
            }
        }
        // The engine's own encodes and presentation.
        engine.set_references(r);
        const auto text = engine.prompt_embeddings(r.prompt);
        if (mode == "edit_text") {
            // The vision tower's merged tokens and DeepStack features, both images in one tensor, one after the other.
            const auto merger = read_f32(ref / "ed_vis_merger.f32");
            std::vector<std::vector<float>> deep;
            for (uint32_t d = 0; d < mako::models::QwenVision::kDeepstack; ++d)
                deep.push_back(read_f32(ref / std::format("ed_vis_deepstack{}.f32", d)));
            size_t at = 0;
            for (int i = 0; i < 2; ++i) {
                const auto& f = engine.reference_features(i);
                compare(std::format("vision tower, reference {}", i + 1), f.embeds,
                        std::vector<float>(merger.begin() + at, merger.begin() + at + f.embeds.size()), 5120, 0.02);
                for (uint32_t d = 0; d < mako::models::QwenVision::kDeepstack; ++d)
                    compare(std::format("DeepStack {}, reference {}", d, i + 1), f.deepstack[d],
                            std::vector<float>(deep[d].begin() + at, deep[d].begin() + at + f.embeds.size()), 5120, 0.02);
                at += f.embeds.size();
            }
            check(at == merger.size(), std::format("{} merged vision tokens in all", at / 5120));
            const auto want_tags = tags_of(m["ed_tags"]);
            check(engine.text_tags() == want_tags, std::format("{} presentation rows tagged as the reference's {}",
                                                               engine.text_tags().size(), want_tags.size()));
            // The presentation's ids, built as the engine builds them, against the reference's.
            const auto want_ids = ids_of(m["ed_ids"]);
            std::vector<int32_t> own;
            {
                auto special = [&](const char* n) { return engine.tokenizer().encode(n, true)[0]; };
                for (int i = 0; i < 2; ++i) {
                    const auto& f = engine.reference_features(i);
                    const auto label = engine.tokenizer().encode(std::format("Picture {}: ", i + 1), false);
                    own.insert(own.end(), label.begin(), label.end());
                    own.push_back(special("<|vision_start|>"));
                    own.insert(own.end(), f.rows * f.cols, special("<|image_pad|>"));
                    own.push_back(special("<|vision_end|>"));
                }
                const auto p = engine.tokenizer().encode(r.prompt, true);
                own.insert(own.end(), p.begin(), p.end());
            }
            check(own == want_ids, std::format("{} presentation ids as the reference's {}", own.size(), want_ids.size()));
            compare("edit text embeddings (layer 50)", text, read_f32(ref / "ed_text_embeds.f32"), H3::kText, 0.05);
            return failures ? 1 : 0;
        }
        H3& dit = engine.transformer();
        std::map<std::string, std::vector<float>> probes;
        dit.set_probe([&](const std::string& name, int layer, const float* data, size_t n) {
            probes[layer < 0 ? name : std::format("{}{}", name, layer)] = std::vector<float>(data, data + n);
        });
        // The reference's text embeddings and tags, so the transformer check does not carry the encoder's error.
        engine.set_text_tags(tags_of(m["ed_tags"]));
        engine.begin(read_f32(ref / "ed_text_embeds.f32"), lat, lat);
        const uint32_t C = uint32_t(m["ed_reference_rows"].get_number());
        const auto sig = floats_of(m["sigmas"]);
        const uint32_t steps = uint32_t(m["ed_steps"].get_number());
        // The engine's conditioning rows (its VAE encode of the references, the reference's noises) against the
        // reference's; the step rows carry the reference's.
        compare("reference rows, the engine's", engine.reference_rows(), read_f32(ref / "ed_reference_rows.f32"), 96, 1e-2);
        compare("text after the refiner", probes["txt"], read_f32(ref / "ed_s0_txt.f32"), H3::kDim, 1e-2);
        for (uint32_t i = 0; i < steps; ++i) {
            probes.clear();
            const auto rows = read_f32(ref / std::format("ed_step{}_rows.f32", i));
            if (i == 0)
                compare("reference rows", std::vector<float>(rows.begin(), rows.begin() + uint64_t(C) * 96),
                        read_f32(ref / "ed_reference_rows.f32"), 96, 1e-5);
            const auto vel = engine.step(unpacked(std::span(rows).subspan(uint64_t(C) * 96), lat, lat), sig[i]);
            std::println(" step {} (sigma {})", i, sig[i]);
            {
                // The packed input by segment: the text rows (refined), the references' and the generated rows.
                const auto want_x0 = read_f32(ref / std::format("ed_s{}_x0.f32", i));
                const auto& got_x0 = probes["x0"];
                const uint32_t T = uint32_t(m["ed_text_rows"].get_number());
                if (got_x0.size() == want_x0.size())
                    for (const auto& [name, r0, n] : {std::tuple{"text rows", 0u, T}, {"reference rows", T, C},
                                                      {"generated rows", T + C, N / 4}}) {
                        const Diff d = diff(std::span(got_x0).subspan(uint64_t(r0) * H3::kDim, uint64_t(n) * H3::kDim),
                                            std::vector<float>(want_x0.begin() + uint64_t(r0) * H3::kDim,
                                                               want_x0.begin() + uint64_t(r0 + n) * H3::kDim),
                                            H3::kDim);
                        std::println("        packed input, {}: rel error {:.2e}, lowest row cosine {:.5f}", name, d.rel, d.min_cos);
                    }
            }
            compare("packed input", probes["x0"], read_f32(ref / std::format("ed_s{}_x0.f32", i)), H3::kDim, 5e-3);
            // The engine's VAE encode of the references (3e-3 off the reference's fp32 one) moves the late blocks and
            // the velocity more than the transformer's own error: the exact-row check below has the transformer alone.
            for (int b : {0, 1, 24, 49})
                compare(std::format("block {}", b), probes[std::format("block{}", b)],
                        read_f32(ref / std::format("ed_s{}_block{}.f32", i, b)), H3::kDim, b < 40 ? 2e-2 : 0.15);
            std::vector<float> want = read_f32(ref / std::format("ed_step{}_pred.f32", i));
            want.erase(want.begin(), want.begin() + uint64_t(C) * 96);
            for (float& v : want)
                v = -v;
            compare("velocity of the generated rows", packed(vel, lat, lat), want, 96, 6e-2);
            // The transformer alone on this layout: the reference's rows (its reference rows exactly) straight in,
            // its data-ward velocity negated like the engine's.
            const auto exact = dit.step(rows, {}, 1.0f - sig[i], 0.0f, H3Image::kRefT, i);
            std::vector<float> ev(exact.video.begin() + uint64_t(C) * 96, exact.video.end());
            for (float& v : ev)
                v = -v;
            compare("velocity of the generated rows, from the reference's rows", ev, want, 96, 4e-2);
        }
    } else {
        std::println(stderr, "unknown mode {}", mode);
        return 2;
    }
    if (failures)
        std::println("{} failed", failures);
    return failures ? 1 : 0;
}
