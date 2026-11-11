#include "image/h3_image.hpp"

#include "image/png.hpp"
#include "log/log.hpp"

#include <glaze/glaze.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <random>
#include <stdexcept>

#include <sys/sysctl.h>

namespace mako::image {

namespace {

namespace fs = std::filesystem;

// The sampling grid the checkpoint's heads were distilled on (pdd_config.json), when the model directory has none.
constexpr float kGrid[] = {1.0f, 0.9675236940383911f, 0.8875485062599182f, 0.75f, 0.5327739119529724f,
                           0.23200887441635132f, 0.0f};

std::string read_text(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

// RGBA [h][w][4] in [-1, 1] as 8-bit RGB [h][w][3], resized to W x H first when that differs.
std::vector<uint8_t> rgb8(std::span<const float> rgba, uint32_t w, uint32_t h, uint32_t W, uint32_t H)
{
    std::vector<float> sized;
    if (W != w || H != h) {
        sized = resize_image(rgba, w, h, W, H);
        rgba = sized;
    }
    std::vector<uint8_t> out(uint64_t(W) * H * 3);
    for (uint64_t p = 0; p < uint64_t(W) * H; ++p)
        for (uint32_t c = 0; c < 3; ++c)
            out[p * 3 + c] = uint8_t(std::lround(std::clamp(rgba[p * 4 + c] * 0.5f + 0.5f, 0.0f, 1.0f) * 255.0f));
    return out;
}

} // namespace

H3Image::H3Image(const std::string& dir, mtl::Context& ctx, const Options& opts)
    : Engine(ctx, opts), text_dir_(opts.text_encoder.empty() ? dir + "/text_encoder" : opts.text_encoder)
{
    dir_ = dir;
    lat_ = models::H3Vae::kLatent; // a token is one latent; the transformer takes them as 2x2 patches
    grid_.assign(std::begin(kGrid), std::end(kGrid));
    if (const std::string cfg = dir + "/pdd_config.json"; fs::exists(cfg)) {
        glz::generic j;
        if (glz::read_json(j, read_text(cfg)) || !j.contains("sigmas") || !j["sigmas"].is_array())
            throw std::runtime_error(cfg + ": no sigmas");
        grid_.clear();
        for (const auto& v : j["sigmas"].get_array())
            grid_.push_back(float(v.get_number()));
    }
    if (grid_.size() < 2 || grid_.front() != 1.0f || grid_.back() != 0.0f)
        throw std::runtime_error("the schedule must run from 1 to 0");
    // The released sampler rounds every sigma through its float timestep 1 - sigma and steps on the result.
    for (float& s : grid_)
        s = 1.0f - (1.0f - s);
    load_tokenizer(fs::exists(dir + "/tokenizer/tokenizer.json") ? dir + "/tokenizer/tokenizer.json"
                                                                  : dir + "/processor/tokenizer.json");
    // Layer 50's output conditions the transformer: the layers past it are not loaded.
    load_text(text_dir_, models::QwenText::Options{.layers = kTextLayer});
    if (text_width() != models::H3::kText)
        throw std::runtime_error(std::format("text encoder: {} wide, not {}", text_width(), models::H3::kText));
    // The release's vae/ (the model's own tensor names, as the ComfyUI repack's fp16 copy has them); vae_diffusers/
    // holds the same weights under diffusers' names, which are not read.
    const std::string vcfg = dir + "/vae/config.json";
    std::string vw = opts.vae;
    if (vw.empty() && fs::exists(dir + "/vae/diffusion_pytorch_model.safetensors"))
        vw = dir + "/vae/diffusion_pytorch_model.safetensors";
    if (vw.empty())
        throw std::runtime_error(dir + ": no VAE weights (vae/diffusion_pytorch_model.safetensors, or \"vae\" in mako.json)");
    if (fs::is_directory(vw))
        vw += "/diffusion_pytorch_model.safetensors";
    vae_ = std::make_unique<models::H3Vae>(vw, vcfg, ctx, opts.telemetry, "image VAE");
    if (vae_->patch_t() != 1)
        throw std::runtime_error(std::format("{}: a decoder with tokens {} frames deep (the video model's VAE)", vw, vae_->patch_t()));
    dit_path_ = opts.transformer.empty() ? dir + "/transformer" : opts.transformer;
    if (!fs::exists(dit_path_))
        throw std::runtime_error(dit_path_ + ": no transformer");
    // The release's shards keep the per-step heads in a file beside them; single files carry them.
    if (const std::string heads = dir + "/pdd_fused_heads.safetensors"; fs::is_directory(dit_path_) && fs::exists(heads))
        heads_path_ = heads;
    // The weight files, the text encoder's (its 50 of 64 layers) and the transformer's, against the machine's memory.
    auto bytes = [](const std::string& p) {
        uint64_t n = 0;
        std::error_code ec;
        if (fs::is_regular_file(p, ec))
            return uint64_t(fs::file_size(p, ec));
        for (const auto& e : fs::recursive_directory_iterator(p, ec))
            if (e.is_regular_file(ec) && e.path().extension() == ".safetensors")
                n += e.file_size(ec);
        return n;
    };
    const double te = double(bytes(text_dir_)) * kTextLayer / 64.0, t = double(bytes(dit_path_));
    uint64_t mem = 0;
    size_t len = sizeof(mem);
    if (sysctlbyname("hw.memsize", &mem, &len, nullptr, 0) != 0)
        mem = 0;
    keeps_both_ = double(mem) >= 1.35 * (te + t);
    if (!keeps_both_)
        log::info("the transformer ({:.0f} GB) and the text encoder ({:.0f} GB) do not fit together with room in this "
                  "machine's {:.0f} GB: each is loaded for its part of a request and freed for the other's",
                  t / 1e9, te / 1e9, double(mem) / 1e9);
}

models::H3& H3Image::transformer()
{
    if (dit_)
        return *dit_;
    if (!keeps_both_ && has_text()) {
        free_text();
        vision_.reset();
        log::info("text encoder freed for the transformer");
    }
    dit_ = std::make_unique<models::H3>(dit_path_, ctx_, tel_, 0, heads_path_);
    if (dit_->has_audio())
        throw std::runtime_error(dit_path_ + ": a transformer with an audio stream (MiniMax-H3's video model)");
    if (dit_->heads() != default_steps())
        throw std::runtime_error(std::format("{}: {} output heads for the schedule's {} steps{}", dit_path_,
                                             dit_->heads(), default_steps(),
                                             dit_->heads() == 1 ? " (pdd_fused_heads.safetensors beside the shards)" : ""));
    for (const Lora& l : loras_)
        dit_->add_lora(l.path, l.scale, true);
    return *dit_;
}

std::pair<uint32_t, uint32_t> H3Image::bound_short_edge(uint32_t w, uint32_t h, uint32_t lo, uint32_t hi)
{
    const uint32_t s = std::min(w, h);
    if (lo <= s && s <= hi)
        return {w, h};
    const double scale = double(s < lo ? lo : hi) / double(s);
    auto side = [&](uint32_t v) { return std::max<uint32_t>(32, uint32_t(std::nearbyint(v * scale / 32.0)) * 32); };
    return {side(w), side(h)};
}

std::pair<uint32_t, uint32_t> H3Image::canvas(uint32_t width, uint32_t height, std::optional<uint32_t> min_area) const
{
    // A smaller image than the canvas is generated at the same aspect ratio on the canvas and resized back.
    const double area = min_area ? double(*min_area) : double(kMinArea);
    const double scale = std::max(1.0, std::sqrt(area / (double(width) * height)));
    auto side = [&](uint32_t v) { return std::max<uint32_t>(32, uint32_t(std::nearbyint(v * scale / 32.0)) * 32); };
    return {side(width), side(height)};
}

models::QwenVision& H3Image::vision()
{
    if (!keeps_both_ && dit_) {
        dit_.reset();
        log::info("transformer freed for the text encoder");
    }
    if (!vision_)
        vision_ = std::make_unique<models::QwenVision>(text_dir_, ctx_, tel_);
    return *vision_;
}

void H3Image::set_references(const Request& r)
{
    refs_.clear();
    cond_rows_.clear();
    if (r.references.empty())
        return;
    if (!r.reference_noise.empty() && r.reference_noise.size() != 2 * r.references.size())
        throw std::runtime_error("reference noise: two per reference (the posterior's, then the conditioning's)");
    std::mt19937_64 rng(r.seed ^ 0x6d696e696d617865ull);
    std::normal_distribution<float> nd;
    for (size_t i = 0; i < r.references.size(); ++i) {
        const Request::Reference& src = r.references[i];
        if (!src.width || !src.height || src.rgba.size() != uint64_t(src.width) * src.height * 4)
            throw std::runtime_error(std::format("reference {}: RGBA [height][width][4]", i + 1));
        if (std::min(src.width, src.height) < 32)
            throw std::runtime_error(std::format("reference {}: at least 32 pixels a side", i + 1));
        Ref ref;
        // The text encoder sees it at a 1024 short edge.
        const auto [vw, vh] = bound_short_edge(src.width, src.height, kVisionShortEdge, kVisionShortEdge);
        ref.features = vision().encode(rgb8(src.rgba, src.width, src.height, vw, vh), vh, vw);
        // The VAE at its own size within a 1024 to 2048 short edge, cropped at the top left to whole latents, the
        // latents then to whole patches.
        const auto [ew, eh] = bound_short_edge(src.width, src.height, kRefMinShortEdge, kRefMaxShortEdge);
        const uint32_t W = ew / 16 * 16, H = eh / 16 * 16, lh = H / 16, lw = W / 16;
        std::vector<uint8_t> rgb = rgb8(src.rgba, src.width, src.height, ew, eh);
        if (W != ew || H != eh) {
            std::vector<uint8_t> cropped(uint64_t(W) * H * 3);
            for (uint32_t y = 0; y < H; ++y)
                std::copy_n(&rgb[uint64_t(y) * ew * 3], uint64_t(W) * 3, &cropped[uint64_t(y) * W * 3]);
            rgb = std::move(cropped);
        }
        std::vector<float> noise(uint64_t(lh) * lw * models::H3Vae::kLatent), mix(noise.size());
        for (auto [dst, k] : {std::pair{&noise, 2 * i}, std::pair{&mix, 2 * i + 1}}) {
            if (!r.reference_noise.empty()) {
                if (r.reference_noise[k].size() != dst->size())
                    throw std::runtime_error(std::format("reference {} noise: [{}][{}][24]", i + 1, lh, lw));
                std::ranges::copy(r.reference_noise[k], dst->begin());
            } else {
                for (float& v : *dst)
                    v = nd(rng);
            }
        }
        const auto z = vae_->encode(rgb, H, W, noise);
        // Rows of 2x2 patches ([c][dy][dx]) of the even grid, noised to kRefT and held there.
        ref.lh = lh / 2 * 2;
        ref.lw = lw / 2 * 2;
        const float t = kRefT, u = float(1.0 - double(kRefT));
        for (uint32_t py = 0; py < ref.lh / 2; ++py)
            for (uint32_t px = 0; px < ref.lw / 2; ++px)
                for (uint32_t c = 0; c < models::H3Vae::kLatent; ++c)
                    for (uint32_t dy = 0; dy < 2; ++dy)
                        for (uint32_t dx = 0; dx < 2; ++dx) {
                            const uint64_t v = (uint64_t(2 * py + dy) * lw + 2 * px + dx) * models::H3Vae::kLatent + c;
                            cond_rows_.push_back(t * z[v] + u * mix[v]);
                        }
        refs_.push_back(std::move(ref));
    }
}

std::vector<float> H3Image::prompt_embeddings(const std::string& prompt)
{
    if (!has_text()) {
        if (dit_) {
            dit_.reset();
            log::info("transformer freed for the text encoder");
        }
        load_text(text_dir_, models::QwenText::Options{.layers = kTextLayer});
    }
    auto special = [&](const char* name) {
        const auto v = tok_->encode(name, true);
        if (v.size() != 1)
            throw std::runtime_error(std::format("the tokenizer has no {} token", name));
        return v[0];
    };
    std::vector<int32_t> ids;
    std::vector<uint8_t> tags;
    std::vector<models::QwenText::Vision> blocks;
    for (size_t i = 0; i < refs_.size(); ++i) {
        // The label carries no angle brackets, unlike the <Picture i> the prompt refers to it by.
        auto label = tok_->encode(std::format("Picture {}: ", i + 1), false);
        if (label.size() > kMaxTokens)
            label.resize(kMaxTokens);
        ids.insert(ids.end(), label.begin(), label.end());
        tags.insert(tags.end(), label.size(), uint8_t(models::H3::kTextTag));
        const models::QwenVision::Features& f = refs_[i].features;
        const uint32_t n = f.rows * f.cols;
        ids.push_back(special("<|vision_start|>"));
        models::QwenText::Vision b;
        b.first = uint32_t(ids.size());
        b.rows = f.rows;
        b.cols = f.cols;
        b.embeds = f.embeds;
        for (const auto& d : f.deepstack)
            b.deepstack.emplace_back(d);
        blocks.push_back(std::move(b));
        ids.insert(ids.end(), n, special("<|image_pad|>"));
        ids.push_back(special("<|vision_end|>"));
        tags.insert(tags.end(), n + 2, uint8_t(models::H3::kVideoTag));
    }
    auto p = tok_->encode(prompt, true);
    if (p.size() > kMaxTokens) // cut at the end, as the reference does
        p.resize(kMaxTokens);
    ids.insert(ids.end(), p.begin(), p.end());
    tags.insert(tags.end(), p.size(), uint8_t(models::H3::kTextTag));
    if (ids.empty())
        throw std::runtime_error("the prompt has no tokens");
    text_tags_ = std::move(tags);
    const uint32_t taps[1] = {kTextLayer};
    return encode_text(ids, 0, taps, false, blocks);
}

void H3Image::begin(std::span<const float> prompt, uint32_t h, uint32_t w, uint32_t slot)
{
    if (slot)
        throw std::runtime_error("MiniMax-H3-Image is not guided: no negative prompt");
    if (prompt.empty() || prompt.size() % models::H3::kText)
        throw std::runtime_error("prompt embeddings must be rows of 5120");
    if (h % 2 || w % 2)
        throw std::runtime_error("the latent grid must be even (sizes multiples of 32)");
    models::H3::Layout l;
    l.text = uint32_t(prompt.size() / models::H3::kText);
    l.frames = 1;
    l.h = h;
    l.w = w;
    if (text_tags_.size() == l.text)
        l.text_tags = text_tags_;
    else if (!refs_.empty())
        throw std::runtime_error("the text rows' tags do not match the embeddings (prompt_embeddings() makes both)");
    for (const Ref& r : refs_)
        l.refs.push_back({models::H3::Layout::Reference::Kind::image, 1, r.lh, r.lw, 0});
    if (cond_rows_.size() != uint64_t(l.cond_rows()) * models::H3::kVideoPatch)
        throw std::logic_error("reference rows");
    transformer().begin(prompt, l);
}

std::vector<float> H3Image::step(std::span<const float> latents, float sigma, uint32_t slot)
{
    if (slot)
        throw std::runtime_error("MiniMax-H3-Image is not guided: no negative prompt");
    if (!dit_)
        throw std::runtime_error("begin() first");
    const auto& l = dit_->layout();
    const uint32_t N = l.frames * l.rows_per_frame(), C = l.cond_rows(), P = models::H3::kVideoPatch;
    const uint32_t lh = l.h, lw = l.w, ph = lh / 2, pw = lw / 2;
    if (latents.size() != uint64_t(lh) * lw * lat_)
        throw std::runtime_error(std::format("latents must be [h * w][{}]", lat_));
    // The step's head: the grid interval it crosses, or across several (fewer steps than the grid's) their heads
    // blended by the sigma each covers; the grid's nearest points to the step's two levels bound it. The step's
    // next level comes from the schedule sigmas() made (the grid itself when none was asked).
    const auto& sched = std::ranges::find(schedule_, sigma) != schedule_.end() ? schedule_ : grid_;
    const auto at = std::ranges::find(sched, sigma);
    if (at == sched.end() || at + 1 == sched.end())
        throw std::runtime_error(std::format("sigma {} is not a step of the schedule", sigma));
    const uint32_t n = default_steps();
    auto nearest = [&](float s) {
        uint32_t best = 0;
        for (uint32_t k = 1; k <= n; ++k)
            if (std::abs(grid_[k] - s) < std::abs(grid_[best] - s))
                best = k;
        return best;
    };
    const uint32_t start = std::min(nearest(sigma), n - 1), stop = std::max(nearest(*(at + 1)), start + 1);
    uint32_t head = start;
    if (stop > start + 1) {
        std::vector<float> w(n, 0.0f);
        float sum = 0;
        for (uint32_t k = start; k < stop; ++k) {
            w[k] = grid_[k] - grid_[k + 1];
            sum += w[k];
        }
        for (float& v : w)
            v /= sum;
        head = dit_->blend_heads(w);
    }
    // The latents as rows of 2x2 patches ([c][dy][dx]) after the references'.
    std::vector<float> video(cond_rows_);
    video.resize(uint64_t(C + N) * P);
    for (uint32_t py = 0; py < ph; ++py)
        for (uint32_t px = 0; px < pw; ++px)
            for (uint32_t c = 0; c < lat_; ++c)
                for (uint32_t dy = 0; dy < 2; ++dy)
                    for (uint32_t dx = 0; dx < 2; ++dx)
                        video[(uint64_t(C) + py * pw + px) * P + (c * 2 + dy) * 2 + dx] =
                            latents[(uint64_t(2 * py + dy) * lw + 2 * px + dx) * lat_ + c];
    const auto vel = dit_->step(video, {}, 1.0f - sigma, 0.0f, kRefT, head);
    // The transformer's velocity points at the data; the loop steps the latents along the noise's.
    std::vector<float> out(latents.size());
    for (uint32_t py = 0; py < ph; ++py)
        for (uint32_t px = 0; px < pw; ++px)
            for (uint32_t c = 0; c < lat_; ++c)
                for (uint32_t dy = 0; dy < 2; ++dy)
                    for (uint32_t dx = 0; dx < 2; ++dx)
                        out[(uint64_t(2 * py + dy) * lw + 2 * px + dx) * lat_ + c] =
                            -vel.video[(uint64_t(C) + py * pw + px) * P + (c * 2 + dy) * 2 + dx];
    return out;
}

std::vector<float> H3Image::sigmas(uint32_t steps, uint32_t, std::optional<float> terminal) const
{
    if (terminal && *terminal != 0.0f)
        throw std::runtime_error("MiniMax-H3-Image: the schedule ends at 0 (no terminal)");
    const uint32_t n = default_steps();
    if (!steps)
        steps = n;
    std::vector<float> s;
    if (steps == n) {
        s = grid_;
    } else {
        // The distilled grid interpolated linearly in its index (as ComfyUI's samplers see this model): a step
        // inside one of its intervals takes that interval's head, a step across several a blend of theirs.
        for (uint32_t i = 0; i <= steps; ++i) {
            const double pos = double(n) * i / steps;
            const uint32_t k = std::min(uint32_t(pos), n - 1);
            const double f = pos - k;
            s.push_back(float(grid_[k] * (1.0 - f) + grid_[k + 1] * f));
        }
        s.back() = 0.0f;
    }
    schedule_ = s;
    return s;
}

std::vector<float> H3Image::decode(std::span<const float> latents, uint32_t h, uint32_t w)
{
    if (latents.size() != uint64_t(h) * w * lat_)
        throw std::runtime_error(std::format("latents must be [h * w][{}]", lat_));
    const auto rgb = vae_->decode(latents, 1, h, w);
    // 8-bit levels by truncation, as the released pipeline quantizes its images.
    const uint64_t px = uint64_t(h) * w * 256;
    std::vector<float> out(px * 4);
    for (uint64_t p = 0; p < px; ++p) {
        for (uint32_t c = 0; c < 3; ++c)
            out[p * 4 + c] = std::floor(std::min(rgb[p * 3 + c] * 255.0f, 255.0f)) / 255.0f * 2.0f - 1.0f;
        out[p * 4 + 3] = 1.0f;
    }
    return out;
}

std::vector<float> H3Image::encode(std::span<const float>, uint32_t, uint32_t)
{
    throw std::runtime_error("MiniMax-H3-Image takes reference images, not an image to start from");
}

uint32_t H3Image::add_lora(const Lora& l)
{
    if (l.merge == false)
        throw std::runtime_error("MiniMax-H3-Image keeps no LoRA unmerged");
    if (!std::filesystem::exists(l.path))
        throw std::runtime_error(l.path + ": no such LoRA file");
    loras_.push_back(l);
    // merged into the transformer as it loads, else now
    return dit_ ? dit_->add_lora(l.path, l.scale, true) : 0;
}

} // namespace mako::image
