#include "image/engine.hpp"

#include "image/anima.hpp"
#include "image/er_sde.hpp"
#include "image/h3_image.hpp"

#include "log/log.hpp"
#include "gguf/read_pool.hpp"
#include "image/krea2.hpp"
#include "image/moss_image.hpp"
#include "image/qwen_image.hpp"
#include "image/sdxl.hpp"

#include <glaze/glaze.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <map>
#include <random>
#include <stdexcept>

namespace mako::image {

namespace {

struct UpArgs {
    uint32_t h, w, c, c_out, ft, repeats;
};
struct TransposeArgs {
    uint32_t rows, cols, ldx, col0;
};

safetensors::Files open_files(const std::string& path)
{
    auto f = safetensors::Files::open(path);
    if (!f)
        throw std::runtime_error(f.error());
    return std::move(*f);
}

glz::generic read_json_file(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("cannot open " + path);
    const std::string text{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    glz::generic j;
    if (glz::read_json(j, text) || !j.is_object())
        throw std::runtime_error(path + ": not a JSON object");
    return j;
}

// Round to nearest even BF16.
// The first Qwen-Image VAE's names in its original layout (as single-file checkpoints have it) for the Diffusers
// layout's.
std::string wan_name(const std::string& n)
{
    static constexpr std::pair<const char*, const char*> kRes[] = {
        {"norm1.", "residual.0."}, {"conv1", "residual.2"}, {"norm2.", "residual.3."}, {"conv2", "residual.6"},
        {"conv_shortcut", "shortcut"}};
    auto res = [](const std::string& rest) {
        for (const auto& [d, o] : kRes)
            if (rest.starts_with(d))
                return o + rest.substr(std::strlen(d));
        return rest;
    };
    static constexpr std::pair<const char*, const char*> kTop[] = {
        {"post_quant_conv", "conv2"}, {"decoder.conv_in", "decoder.conv1"}, {"decoder.norm_out.", "decoder.head.0."},
        {"decoder.conv_out", "decoder.head.2"}, {"decoder.mid_block.attentions.0.", "decoder.middle.1."}};
    for (const auto& [d, o] : kTop)
        if (n.starts_with(d))
            return o + n.substr(std::strlen(d));
    if (n.starts_with("decoder.mid_block.resnets."))
        return std::format("decoder.middle.{}.", n[26] == '0' ? 0 : 2) + res(n.substr(28));
    unsigned b = 0, j = 0;
    int used = 0;
    if (std::sscanf(n.c_str(), "decoder.up_blocks.%u.resnets.%u.%n", &b, &j, &used) == 2 && used > 0)
        return std::format("decoder.upsamples.{}.", b * 4 + j) + res(n.substr(size_t(used)));
    for (const char* up : {"decoder.up_blocks.%u.upsamplers.0.%n", "decoder.up_blocks.%u.upsampler.%n"})
        if (used = 0; std::sscanf(n.c_str(), up, &b, &used) == 1 && used > 0)
            return std::format("decoder.upsamples.{}.", b * 4 + 3) + n.substr(size_t(used));
    return n;
}

// The original AutoencoderKL decoder names (single-file SDXL checkpoints) for the Diffusers layout's; up block i is
// the original's levels - 1 - i.
std::string ldm_name(const std::string& n, unsigned levels)
{
    auto res = [](const std::string& rest) {
        return rest.starts_with("conv_shortcut") ? "nin_shortcut" + rest.substr(13) : rest;
    };
    static constexpr std::pair<const char*, const char*> kTop[] = {
        {"decoder.conv_norm_out.", "decoder.norm_out."},
        {"decoder.mid_block.resnets.0.", "decoder.mid.block_1."},
        {"decoder.mid_block.resnets.1.", "decoder.mid.block_2."},
        {"decoder.mid_block.attentions.0.group_norm.", "decoder.mid.attn_1.norm."},
        {"decoder.mid_block.attentions.0.to_q", "decoder.mid.attn_1.q"},
        {"decoder.mid_block.attentions.0.to_k", "decoder.mid.attn_1.k"},
        {"decoder.mid_block.attentions.0.to_v", "decoder.mid.attn_1.v"},
        {"decoder.mid_block.attentions.0.to_out.0", "decoder.mid.attn_1.proj_out"}};
    for (const auto& [d, o] : kTop)
        if (n.starts_with(d))
            return o + res(n.substr(std::strlen(d)));
    unsigned b = 0, j = 0;
    int used = 0;
    if (std::sscanf(n.c_str(), "decoder.up_blocks.%u.resnets.%u.%n", &b, &j, &used) == 2 && used > 0 && b < levels)
        return std::format("decoder.up.{}.block.{}.", levels - 1 - b, j) + res(n.substr(size_t(used)));
    if (used = 0; std::sscanf(n.c_str(), "decoder.up_blocks.%u.upsamplers.0.%n", &b, &used) == 1 && used > 0 &&
                  b < levels)
        return std::format("decoder.up.{}.upsample.", levels - 1 - b) + n.substr(size_t(used));
    return n;
}

// The first Qwen-Image VAE's latent statistics, where its file comes without its config.
constexpr float kWanMean[16] = {-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                                0.4134,  -0.0715, 0.5517,  -0.3632, -0.1922, -0.9497, 0.2503, -0.2921};
constexpr float kWanStd[16] = {2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
                               3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.916};

} // namespace

std::optional<Sampler> parse_sampler(std::string_view name)
{
    if (name == "euler")
        return Sampler::euler;
    if (name == "euler_a" || name == "euler_ancestral")
        return Sampler::euler_ancestral;
    if (name == "er_sde")
        return Sampler::er_sde;
    return std::nullopt;
}

const char* sampler_name(Sampler s)
{
    return s == Sampler::euler ? "euler" : s == Sampler::euler_ancestral ? "euler_a" : "er_sde";
}

void Engine::set_skip_blocks(std::span<const uint32_t> blocks)
{
    if (!blocks.empty())
        throw std::runtime_error(std::format("{}: no blocks to skip", name()));
}

std::string Engine::family(const std::string& dir)
{
    if (!std::filesystem::exists(dir + "/model_index.json"))
        return Anima::detect(dir) ? "Anima" : !Sdxl::checkpoint(dir).empty() ? "SDXL" : "";
    try {
        const auto j = read_json_file(dir + "/model_index.json");
        const std::string cls = j.contains("_class_name") && j["_class_name"].is_string() ? j["_class_name"].get_string()
                                                                                         : "";
        return cls == "QwenImage21Pipeline" ? "Qwen-Image 2.1"
               : cls == "Krea2Pipeline"     ? "Krea 2"
               : cls == "MOSSImagePipeline" ? "MOSS-Image"
               : cls == "MiniMaxH3ImagePipeline" ? "MiniMax-H3-Image"
                                                  : "";
    } catch (const std::exception&) {
        return "";
    }
}

std::unique_ptr<Engine> Engine::open(const std::string& dir, mtl::Context& ctx, const Options& opts)
{
    // Single-file parts in their own directories (diffusion_models/, text_encoders/, vae/): Anima's layout.
    if (!std::filesystem::exists(dir + "/model_index.json") && Anima::detect(dir))
        return std::make_unique<Anima>(dir, ctx, opts);
    // A single-file SDXL checkpoint.
    if (!std::filesystem::exists(dir + "/model_index.json") && !Sdxl::checkpoint(dir).empty())
        return std::make_unique<Sdxl>(dir, ctx, opts);
    const auto j = read_json_file(dir + "/model_index.json");
    const std::string cls = j.contains("_class_name") && j["_class_name"].is_string() ? j["_class_name"].get_string() : "";
    if (cls == "QwenImage21Pipeline")
        return std::make_unique<QwenImage>(dir, ctx, opts);
    if (cls == "Krea2Pipeline")
        return std::make_unique<Krea2>(dir, ctx, opts);
    if (cls == "MOSSImagePipeline")
        return std::make_unique<MossImage>(dir, ctx, opts);
    if (cls == "MiniMaxH3ImagePipeline")
        return std::make_unique<H3Image>(dir, ctx, opts);
    if (!opts.adapter.empty())
        throw std::runtime_error(std::format("{}: adapters are for MOSS-Image", dir));
    if (!opts.control.empty())
        throw std::runtime_error(std::format("{}: control adapters are for MOSS-Image", dir));
    throw std::runtime_error(std::format("{}: an image model mako does not know ({})", dir, cls.empty() ? "?" : cls));
}

Engine::Parts Engine::diffusers_parts(const std::string& dir, const Options& opts)
{
    Parts p;
    p.text_encoder = opts.text_encoder.empty() ? dir + "/text_encoder" : opts.text_encoder;
    p.vae = dir + "/vae";
    p.scheduler = dir + "/scheduler/scheduler_config.json";
    p.tokenizer = std::filesystem::exists(dir + "/processor/tokenizer.json") ? dir + "/processor/tokenizer.json"
                                                                            : dir + "/tokenizer/tokenizer.json";
    return p;
}

Engine::Engine(const std::string& dir, mtl::Context& ctx, const Options& opts)
    : Engine(diffusers_parts(dir, opts), ctx, opts)
{
    dir_ = dir;
}

Engine::Engine(const Parts& parts, mtl::Context& ctx, const Options& opts)
    : models::ConvRuntime(ctx, opts.telemetry), vae_files_(open_files(parts.vae))
{
    load_text(parts.text_encoder, models::QwenText::Options{.theta = parts.text_theta, .layers = parts.text_layers});

    wan_ = vae_files_.find("decoder.conv1.weight") && !vae_files_.find("decoder.conv_in.weight");
    const std::string vcfg = parts.vae + "/config.json";
    const auto vj = std::filesystem::exists(vcfg) ? read_json_file(vcfg) : glz::generic{};
    kl_ = vj.contains("_class_name") && vj["_class_name"].is_string() &&
          vj["_class_name"].get_string() == "AutoencoderKLFlux2";
    if (kl_) {
        // Tokens pack a 2x2 patch of the VAE's channels; its batch norm's statistics are per packed value.
        load_vae_kl();
        z_ = post_quant_.cin;
        patch_ = 2;
        lat_ = z_ * 4;
        double eps = 1e-4;
        if (vj.contains("batch_norm_eps") && vj["batch_norm_eps"].is_number())
            eps = vj["batch_norm_eps"].get_number();
        lat_mean_ = floats(vae_files_, "bn.running_mean");
        const auto var = floats(vae_files_, "bn.running_var");
        if (lat_mean_.size() != lat_ || var.size() != lat_)
            throw std::runtime_error(std::format("vae: batch norm over {} values, not {}", var.size(), lat_));
        for (float v : var)
            lat_std_.push_back(std::sqrt(v + float(eps)));
    } else {
        load_vae();
        std::vector<float> mean, sd;
        if (vj.contains("latents_mean") && vj.contains("latents_std")) {
            for (const auto& v : vj["latents_mean"].get_array())
                mean.push_back(float(v.get_number()));
            for (const auto& v : vj["latents_std"].get_array())
                sd.push_back(float(v.get_number()));
        } else if (wan_) {
            mean.assign(std::begin(kWanMean), std::end(kWanMean));
            sd.assign(std::begin(kWanStd), std::end(kWanStd));
        } else {
            throw std::runtime_error("vae: no config.json with latents_mean / latents_std");
        }
        z_ = uint32_t(post_quant_.cin);
        patch_ = z_ == kLatent ? 1 : z_ * 4 == kLatent ? 2 : 0;
        if (!patch_ || mean.size() != z_ || sd.size() != z_)
            throw std::runtime_error(std::format("vae: {} latent channels, {} means, {} deviations", z_, mean.size(),
                                                 sd.size()));
        // Per token value: value (c * p + py) * p + px is channel c's.
        for (uint32_t i = 0; i < kLatent; ++i) {
            lat_mean_.push_back(mean[i / (patch_ * patch_)]);
            lat_std_.push_back(sd[i / (patch_ * patch_)]);
        }
        residual_ = vj.contains("is_residual") && vj["is_residual"].is_boolean() && vj["is_residual"].get_boolean();
    }

    const auto sj = std::filesystem::exists(parts.scheduler) ? read_json_file(parts.scheduler) : glz::generic{};
    auto num = [&](const char* key, double& v) {
        if (sj.contains(key) && sj[key].is_number())
            v = sj[key].get_number();
    };
    num("base_image_seq_len", sched_.base_seq);
    num("max_image_seq_len", sched_.max_seq);
    num("base_shift", sched_.base_shift);
    num("max_shift", sched_.max_shift);
    num("shift_terminal", sched_.terminal);

    load_tokenizer(parts.tokenizer);
}

Engine::Engine(mtl::Context& ctx, const Options& opts) : models::ConvRuntime(ctx, opts.telemetry) {}

void Engine::load_text(const std::string& dir, const models::QwenText::Options& opts)
{
    text_ = std::make_unique<models::QwenText>(dir, ctx_, tel_, opts);
}

void Engine::load_tokenizer(const std::string& path)
{
    auto t = text::Tokenizer::load_json(path);
    if (!t)
        throw std::runtime_error(t.error());
    tok_ = std::move(*t);
}

Engine::Tensor Engine::matrix(const safetensors::Files& files, const std::string& name)
{
    const auto r = files.find(name);
    if (!r || r.tensor->dims.size() != 2)
        throw std::runtime_error("missing matrix " + name);
    if (r.tensor->type != gguf::Type::F16)
        return upload(files, name, true);
    Tensor t = reserve(r.tensor->bytes, gguf::Type::F16, r.tensor->dims);
    gguf::read_exact(r.file->fd(), static_cast<char*>(t.buf->contents()) + t.off, r.tensor->bytes, r.tensor->offset);
    return t;
}

void Engine::load_vae_ldm(const std::string& file, const std::string& prefix, float scale)
{
    vae_files_ = open_files(file);
    ldm_ = prefix;
    while (vae_files_.find(std::format("{}decoder.up.{}.block.0.conv1.weight", ldm_, ldm_levels_)))
        ++ldm_levels_;
    kl_ = true;
    load_vae_kl();
    z_ = post_quant_.cin;
    patch_ = 2;
    lat_ = z_ * 4;
    lat_mean_.assign(lat_, 0.0f);
    lat_std_.assign(lat_, 1.0f / scale);
}

std::vector<float> Engine::encode_text(std::span<const int32_t> ids, uint32_t drop, std::span<const uint32_t> taps,
                                       bool normed)
{
    if (!text_)
        throw std::runtime_error(std::format("{}: the text encoder is not loaded", name()));
    return text_->encode(ids, drop, taps, normed);
}

std::vector<float> Engine::encode_text(std::span<const int32_t> ids, uint32_t drop, std::span<const uint32_t> taps,
                                       bool normed, std::span<const models::QwenText::Vision> vision)
{
    if (!text_)
        throw std::runtime_error(std::format("{}: the text encoder is not loaded", name()));
    return text_->encode(ids, drop, taps, normed, vision);
}

double Engine::shift_mu(uint32_t tokens) const
{
    const double m = (sched_.max_shift - sched_.base_shift) / (sched_.max_seq - sched_.base_seq);
    return sched_.mu ? *sched_.mu : double(tokens) * m + (sched_.base_shift - m * sched_.base_seq);
}

float Engine::top_sigma(uint32_t tokens) const
{
    const double mu = shift_mu(tokens), t = 1 - 1e-4;
    return float(std::exp(mu) / (std::exp(mu) + (1.0 / t - 1.0)));
}

std::vector<float> Engine::sigmas(uint32_t steps, uint32_t tokens, std::optional<float> terminal) const
{
    const double term = terminal ? double(*terminal) : sched_.terminal;
    const double mu = shift_mu(tokens);
    std::vector<float> s(steps);
    for (uint32_t i = 0; i < steps; ++i) {
        // linspace(1, 1 / steps, steps), then exp(mu) / (exp(mu) + 1 / t - 1), in float as the scheduler does.
        const float t = steps == 1 ? 1.0f : float(1.0 + (1.0 / steps - 1.0) * double(i) / double(steps - 1));
        s[i] = float(std::exp(mu) / (std::exp(mu) + (1.0 / double(t) - 1.0)));
    }
    // Nothing to stretch when the last sigma is 1 (one step): the scale would be 0.
    if (term > 0 && steps > 0 && s.back() < 1.0f) {
        const float scale = (1.0f - s.back()) / (1.0f - float(term));
        for (float& v : s)
            v = 1.0f - (1.0f - v) / scale;
    }
    s.push_back(0.0f);
    return s;
}

std::vector<float> Engine::generate(const Request& r)
{
    if (r.width % 32 || r.height % 32 || !r.width || !r.height)
        throw std::runtime_error("width and height must be multiples of 32");
    const auto [cw, ch] = canvas(r.width, r.height, r.min_area);
    if (cw % 32 || ch % 32 || !cw || !ch)
        throw std::logic_error("the canvas must be multiples of 32");
    const uint32_t h = ch / 16, w = cw / 16, N = h * w;
    // Generated on a larger canvas: resampled to the size asked for.
    auto sized = [&](std::vector<float> img) {
        return cw == r.width && ch == r.height ? img : resize_image(img, cw, ch, r.width, r.height);
    };
    const DataSpace* ds = data_space();
    if (ds && r.sampler)
        throw std::runtime_error(std::format("{}: the adapter has its own sampler", name()));
    if (ds && r.guidance && *r.guidance != 1.0f)
        throw std::runtime_error(std::format("{}: the adapter runs without guidance", name()));
    if (ds && !r.image.empty())
        throw std::runtime_error(std::format("{}: the adapter's sampler starts from noise only", name()));
    if (!r.image.empty() && (r.image.size() != uint64_t(r.width) * r.height * 4 || !(r.strength > 0) ||
                             r.strength > 1))
        throw std::runtime_error("image to image: an RGBA image of the output's size, and a strength in (0, 1]");
    if ((!r.control.color.empty() || !r.control.depth.empty()) && !has_control())
        throw std::runtime_error(std::format("{}: no control adapter for the maps", name()));
    const Sampler sampler = r.sampler.value_or(default_sampler());
    if (sampler == Sampler::euler_ancestral && !predicts_noise())
        throw std::runtime_error(std::format("{}: Euler a needs a model that predicts the noise", name()));
    const auto skip = r.skip_blocks.value_or(default_skip_blocks());
    set_skip_blocks(skip);
    set_references(r);
    const auto prompt = prompt_embeddings(r.prompt);
    std::vector<float> x(uint64_t(N) * lat_);
    std::mt19937_64 rng(r.seed);
    std::normal_distribution<float> nd;
    if (!r.noise.empty()) {
        if (r.noise.size() != x.size())
            throw std::runtime_error(std::format("noise must be [h * w][{}]", lat_));
        std::copy(r.noise.begin(), r.noise.end(), x.begin());
    } else {
        for (float& v : x)
            v = nd(rng);
    }
    if (ds) {
        const uint32_t steps = r.steps ? r.steps : default_steps();
        if (steps > ds->codes.size())
            throw std::runtime_error(std::format("the adapter has {} steps", ds->codes.size()));
        begin(prompt, h, w);
        const uint32_t L = lat_;
        std::vector<float> x0(x.size()), in(x.size());
        for (uint32_t i = 0; i < steps; ++i) {
            for (uint32_t t = 0; t < N; ++t)
                for (uint32_t o = 0; o < L; ++o) {
                    float s = 0;
                    for (uint32_t c = 0; c < L; ++c)
                        s += ds->q[o * L + c] * x0[uint64_t(t) * L + c];
                    in[uint64_t(t) * L + o] = x[uint64_t(t) * L + o] + s;
                }
            const auto v = step(in, ds->codes[i] / 1000.0f);
            for (size_t j = 0; j < x0.size(); ++j)
                x0[j] = x[j] - v[j];
            if (r.progress)
                r.progress(i + 1, steps);
        }
        return sized(decode(x0, h, w));
    }
    auto sig = sigmas(r.steps ? r.steps : default_steps(), N, r.terminal);
    if (!r.image.empty()) {
        // The last steps, from the image noised to the first one's level.
        sig.erase(sig.begin(), sig.begin() + start_step(uint32_t(sig.size() - 1), r.strength));
        const auto z = encode(r.image, h, w);
        const float s0 = sig[0];
        for (size_t j = 0; j < x.size(); ++j)
            x[j] = predicts_noise() ? z[j] + s0 * x[j] : (1 - s0) * z[j] + s0 * x[j];
    }
    const uint32_t steps = uint32_t(sig.size() - 1);
    if (has_control()) {
        auto map = [&](std::span<const float> m) {
            if (m.empty())
                return std::vector<float>{};
            if (m.size() != uint64_t(r.width) * r.height * 4)
                throw std::runtime_error("control: maps must be RGBA of the output's size");
            return encode(m, h, w);
        };
        set_control(map(r.control.color), map(r.control.depth));
    }
    // Noise at the schedule's top level, as the reference scales it there: sqrt(1 + sigma^2) (the image's own unit
    // variance with the noise's).
    if (predicts_noise() && r.image.empty()) {
        const float s0 = std::sqrt(1.0f + sig[0] * sig[0]);
        for (float& v : x)
            v *= s0;
    }
    begin(prompt, h, w);
    const float scale = r.guidance.value_or(default_guidance());
    // At scale 1 the guided velocity is the prompt's own: no negative pass.
    const auto negative = scale == 1.0f ? std::nullopt : r.negative ? r.negative : default_negative();
    if (negative)
        begin(prompt_embeddings(*negative), h, w, 1);
    std::optional<ErSde> er;
    std::vector<float> noise;
    if (sampler == Sampler::er_sde) {
        er.emplace(sig, !predicts_noise(), top_sigma(N));
        noise.resize(x.size());
    }
    for (uint32_t i = 0; i < steps; ++i) {
        const float s = er ? er->sigma(i) : sig[i];
        if (has_control())
            control_on(float(i) < r.control.color_end * float(steps), float(i) < r.control.depth_end * float(steps));
        auto v = step(x, s);
        if (negative) {
            const auto neg = step(x, s, 1);
            for (size_t j = 0; j < v.size(); ++j)
                v[j] = neg[j] + scale * (v[j] - neg[j]);
        }
        if (er) {
            // The image the step sees: the latents less the noise (or velocity) times the level.
            for (size_t j = 0; j < v.size(); ++j)
                v[j] = x[j] - s * v[j];
            if (sig[i + 1] > 0)
                for (float& n : noise)
                    n = nd(rng);
            er->step(i, x, v, noise);
        } else if (sampler == Sampler::euler_ancestral) {
            // Down to the level whose noise, with fresh noise of the remainder added, makes the next one's.
            const double from = sig[i], to = sig[i + 1];
            const double up = std::min(to, std::sqrt(to * to * (from * from - to * to) / (from * from)));
            const float dt = float(std::sqrt(to * to - up * up) - from);
            for (size_t j = 0; j < x.size(); ++j)
                x[j] += dt * v[j];
            if (up > 0)
                for (float& xv : x)
                    xv += float(up) * nd(rng);
        } else {
            const float dt = sig[i + 1] - sig[i];
            for (size_t j = 0; j < x.size(); ++j)
                x[j] += dt * v[j];
        }
        if (r.progress)
            r.progress(i + 1, steps);
    }
    return sized(decode(x, h, w));
}

void Engine::set_control(std::vector<float>, std::vector<float>)
{
    throw std::runtime_error(std::format("{}: takes no control adapter", name()));
}

void Engine::control_on(bool, bool) {}

void Engine::set_references(const Request& r)
{
    if (!r.references.empty())
        throw std::runtime_error(std::format("{}: takes no reference images", name()));
}

std::pair<uint32_t, uint32_t> Engine::canvas(uint32_t width, uint32_t height, std::optional<uint32_t>) const
{
    return {width, height};
}

uint32_t Engine::start_step(uint32_t steps, float strength)
{
    // Below 1, at least the first step is left out: from the top level (pure noise) the image would be lost.
    const uint32_t lo = strength < 1.0f ? 1 : 0;
    return std::clamp(uint32_t(std::lround(steps * (1.0 - strength))), std::min(lo, steps - 1), steps - 1);
}

Provenance Engine::provenance(const Request& r) const
{
    Provenance p;
    p.prompt = r.prompt;
    p.width = r.width;
    p.height = r.height;
    p.seed = r.seed;
    p.steps = r.steps ? r.steps : default_steps();
    if (!r.control.color.empty())
        p.color_end = r.control.color_end;
    if (!r.control.depth.empty())
        p.depth_end = r.control.depth_end;
    if (const auto c = canvas(r.width, r.height, r.min_area); c.first != r.width || c.second != r.height)
        p.canvas = c;
    p.references = uint32_t(r.references.size());
    if (!r.image.empty()) {
        // The share of the steps run, which the strength rounds to.
        const uint32_t all = uint32_t(sigmas(p.steps, (r.width / 16) * (r.height / 16), r.terminal).size() - 1);
        p.strength = float(all - start_step(all, r.strength)) / float(all);
    }
    if (data_space()) {
        p.sampler = "data_space";
        p.guidance = 1;
        return p;
    }
    p.sampler = sampler_name(r.sampler.value_or(default_sampler()));
    p.guidance = r.guidance.value_or(default_guidance());
    if (p.guidance != 1.0f)
        p.negative = r.negative ? r.negative : default_negative();
    if (!p.negative)
        p.guidance = 1.0f;
    else
        p.skip_blocks = r.skip_blocks.value_or(default_skip_blocks());
    return p;
}

Engine::Conv Engine::load_conv(const std::string& name, uint32_t k)
{
    return load_conv(std::span(&name, 1), k);
}

Engine::Conv Engine::load_conv(std::span<const std::string> names, uint32_t k)
{
    std::vector<std::string> full;
    for (const auto& n : names)
        full.push_back(vk(n));
    return load_conv(vae_files_, full, k);
}

std::string Engine::vk(const std::string& name) const
{
    return wan_ ? wan_name(name) : !ldm_.empty() ? ldm_ + ldm_name(name, ldm_levels_) : name;
}

void Engine::load_vae()
{
    auto res = [&](const std::string& p) {
        Res r;
        r.g1 = upload(vae_files_, vk(p + ".norm1.gamma"), false);
        r.g2 = upload(vae_files_, vk(p + ".norm2.gamma"), false);
        r.c1 = load_conv(p + ".conv1", 3);
        r.c2 = load_conv(p + ".conv2", 3);
        if (vae_files_.find(vk(p + ".conv_shortcut.weight"))) {
            r.sc = load_conv(p + ".conv_shortcut", 1);
            r.shortcut = true;
        }
        return r;
    };
    post_quant_ = load_conv("post_quant_conv", 1);
    conv_in_ = load_conv("decoder.conv_in", 3);
    mid0_ = res("decoder.mid_block.resnets.0");
    mid1_ = res("decoder.mid_block.resnets.1");
    attn_g_ = upload(vae_files_, vk("decoder.mid_block.attentions.0.norm.gamma"), false);
    qkv_ = load_conv("decoder.mid_block.attentions.0.to_qkv", 1);
    attn_proj_ = load_conv("decoder.mid_block.attentions.0.proj", 1);
    for (uint32_t i = 0;; ++i) {
        const std::string p = std::format("decoder.up_blocks.{}", i);
        if (!vae_files_.find(vk(p + ".resnets.0.conv1.weight")))
            break;
        Up u;
        for (uint32_t j = 0; vae_files_.find(vk(std::format("{}.resnets.{}.conv1.weight", p, j))); ++j)
            u.res.push_back(res(std::format("{}.resnets.{}", p, j)));
        u.cin = u.res.front().c1.cin;
        u.cout = u.res.back().c2.cout;
        for (const char* up : {".upsampler", ".upsamplers.0"})
            if (vae_files_.find(vk(p + up + ".resample.1.weight"))) {
                u.up = true;
                u.conv = load_conv(p + up + ".resample.1", 3);
                // Blocks that also upsample in time (a time convolution) keep the second of two time copies.
                u.ft = vae_files_.find(vk(p + up + ".time_conv.weight")) ? 2 : 1;
            }
        ups_.push_back(std::move(u));
    }
    out_g_ = upload(vae_files_, vk("decoder.norm_out.gamma"), false);
    conv_out_ = load_conv("decoder.conv_out", 3);
}

Engine::Res Engine::load_res_kl(const std::string& p)
{
    auto vec = [&](const std::string& n) { return upload_f32(vae_files_, vk(n)); };
    Res r;
    r.g1 = vec(p + ".norm1.weight");
    r.b1 = vec(p + ".norm1.bias");
    r.g2 = vec(p + ".norm2.weight");
    r.b2 = vec(p + ".norm2.bias");
    r.c1 = load_conv(p + ".conv1", 3);
    r.c2 = load_conv(p + ".conv2", 3);
    if (vae_files_.find(vk(p + ".conv_shortcut.weight"))) {
        r.sc = load_conv(p + ".conv_shortcut", 1);
        r.shortcut = true;
    }
    return r;
}

void Engine::load_encoder_kl()
{
    if (!kl_ || !ldm_.empty())
        throw std::runtime_error(std::format("{}: no VAE encoder for this model's VAE yet", name()));
    auto e = std::make_unique<VaeEncoder>();
    auto vec = [&](const std::string& n) { return upload_f32(vae_files_, vk(n)); };
    e->in = load_conv("encoder.conv_in", 3);
    for (uint32_t i = 0;; ++i) {
        const std::string p = std::format("encoder.down_blocks.{}", i);
        if (!vae_files_.find(vk(p + ".resnets.0.conv1.weight")))
            break;
        Down d;
        for (uint32_t j = 0; vae_files_.find(vk(std::format("{}.resnets.{}.conv1.weight", p, j))); ++j)
            d.res.push_back(load_res_kl(std::format("{}.resnets.{}", p, j)));
        if (vae_files_.find(vk(p + ".downsamplers.0.conv.weight"))) {
            d.down = true;
            d.conv = load_conv(p + ".downsamplers.0.conv", 3);
            d.conv.stride = 2;
            d.conv.pad = 0;
        }
        e->downs.push_back(std::move(d));
    }
    e->mid0 = load_res_kl("encoder.mid_block.resnets.0");
    e->mid1 = load_res_kl("encoder.mid_block.resnets.1");
    const std::string a = "encoder.mid_block.attentions.0.";
    e->attn_g = vec(a + "group_norm.weight");
    e->attn_b = vec(a + "group_norm.bias");
    const std::string qkv[3] = {a + "to_q", a + "to_k", a + "to_v"};
    e->qkv = load_conv(qkv, 1);
    e->proj = load_conv(a + "to_out.0", 1);
    e->out_g = vec("encoder.conv_norm_out.weight");
    e->out_b = vec("encoder.conv_norm_out.bias");
    e->out = load_conv("encoder.conv_out", 3);
    e->quant = load_conv("quant_conv", 1);
    if (e->quant.cout != 2 * z_)
        throw std::runtime_error(std::format("vae: the encoder gives {} values a pixel, not {}", e->quant.cout, 2 * z_));
    venc_ = std::move(e);
}

void Engine::load_vae_kl()
{
    auto vec = [&](const std::string& n) { return upload_f32(vae_files_, vk(n)); };
    auto res = [&](const std::string& p) { return load_res_kl(p); };
    post_quant_ = load_conv("post_quant_conv", 1);
    conv_in_ = load_conv("decoder.conv_in", 3);
    mid0_ = res("decoder.mid_block.resnets.0");
    mid1_ = res("decoder.mid_block.resnets.1");
    const std::string a = "decoder.mid_block.attentions.0.";
    attn_g_ = vec(a + "group_norm.weight");
    attn_b_ = vec(a + "group_norm.bias");
    const std::string qkv[3] = {a + "to_q", a + "to_k", a + "to_v"};
    qkv_ = load_conv(qkv, 1);
    attn_proj_ = load_conv(a + "to_out.0", 1);
    for (uint32_t i = 0;; ++i) {
        const std::string p = std::format("decoder.up_blocks.{}", i);
        if (!vae_files_.find(vk(p + ".resnets.0.conv1.weight")))
            break;
        Up u;
        for (uint32_t j = 0; vae_files_.find(vk(std::format("{}.resnets.{}.conv1.weight", p, j))); ++j)
            u.res.push_back(res(std::format("{}.resnets.{}", p, j)));
        u.cin = u.res.front().c1.cin;
        u.cout = u.res.back().c2.cout;
        if (vae_files_.find(vk(p + ".upsamplers.0.conv.weight"))) {
            u.up = true;
            u.conv = load_conv(p + ".upsamplers.0.conv", 3);
        }
        ups_.push_back(std::move(u));
    }
    out_g_ = vec("decoder.conv_norm_out.weight");
    out_b_ = vec("decoder.conv_norm_out.bias");
    conv_out_ = load_conv("decoder.conv_out", 3);
}

void Engine::norm_silu(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* x, const Tensor& g, MTL::Buffer* y,
                       uint32_t rows, uint32_t c, const Tensor* b)
{
    if (b)
        return group_norm(enc, x, g, *b, y, rows, c, true);
    // x / ||x|| * sqrt(c) * g over each pixel's channels: an RMS norm (eps as the reference's 1e-12 on the norm).
    norm(enc, x, &g, y, rows, c, c, 1e-24f / float(c));
    unary(enc, "silu", y, rows * c);
}

void Engine::res_block(MTL::ComputeCommandEncoder*& enc, const Res& r, MTL::Buffer* x, uint32_t h, uint32_t w,
                       MTL::Buffer* y, MTL::Buffer* t1, MTL::Buffer* t2)
{
    const uint32_t rows = h * w, cin = r.c1.cin, cout = r.c2.cout;
    norm_silu(enc, x, r.g1, t1, rows, cin, kl_ ? &r.b1 : nullptr);
    conv(enc, r.c1, t1, h, w, t2);
    norm_silu(enc, t2, r.g2, t2, rows, cout, kl_ ? &r.b2 : nullptr);
    conv(enc, r.c2, t2, h, w, y);
    if (r.shortcut) {
        conv(enc, r.sc, x, h, w, t1);
        add(enc, y, t1, y, rows * cout);
    } else {
        add(enc, y, x, y, rows * cout);
    }
}

void Engine::vae_attention(MTL::ComputeCommandEncoder*& enc, const Tensor& g, const Tensor* b, const Conv& qkv,
                           const Conv& proj, MTL::Buffer* x, uint32_t h, uint32_t w, MTL::Buffer* t1, MTL::Buffer* t2,
                           MTL::Buffer* t3)
{
    const uint32_t n = h * w, c = qkv.cin;
    if (b)
        group_norm(enc, x, g, *b, t1, n, c, false);
    else
        norm(enc, x, &g, t1, n, c, c, 1e-24f / float(c));
    conv(enc, qkv, t1, h, w, t2); // [n][3c]: q, k, v
    MTL::Buffer* kc = scratch("v_k", uint64_t(n) * c * 4);
    MTL::Buffer* vt = scratch("v_vt", uint64_t(n) * c * 4);
    // Queries a pass, so the scores stay near 256 MB.
    const uint32_t qn = std::min(n, std::max(256u, uint32_t((64ull << 20) / n) / 256 * 256));
    MTL::Buffer* sc = scratch("v_s", uint64_t(qn) * n * 4);
    const TransposeArgs ka{n, c, 3 * c, c}, va{n, c, 3 * c, 2 * c};
    dispatch(enc, "copy_cols", {t2, kc}, &ka, sizeof(ka), MTL::Size(c, n, 1), MTL::Size(256, 1, 1));
    dispatch(enc, "transpose_f32", {t2, vt}, &va, sizeof(va), MTL::Size(c, n, 1), MTL::Size(256, 1, 1));
    for (uint32_t q0 = 0; q0 < n; q0 += qn) {
        const uint32_t m = std::min(qn, n - q0);
        gemm_->encode(enc, gguf::Type::F32, c, n, m, kc, 0, halves(enc, t2, uint64_t(q0) * 3 * c * 4, m, c, 3 * c), 0,
                      sc, 0);
        after(enc, "vae scores");
        const float sa[2] = {float(n), 1.0f / std::sqrt(float(c))};
        dispatch(enc, "softmax_rows", {sc}, sa, sizeof(sa), MTL::Size(m, 1, 1), MTL::Size(256, 1, 1), false);
        gemm_->encode(enc, gguf::Type::F32, n, c, m, vt, 0, halves(enc, sc, 0, m, n, 0), 0, t1, uint64_t(q0) * c * 4);
        after(enc, "vae pv");
    }
    conv(enc, proj, t1, h, w, t3);
    add(enc, t3, x, x, n * c);
}

std::vector<float> Engine::encode(std::span<const float> rgba, uint32_t h, uint32_t w)
{
    const uint32_t H = 16 * h, W = 16 * w;
    if (rgba.size() != uint64_t(H) * W * 4)
        throw std::runtime_error(std::format("encode: the image must be [{}][{}][4]", H, W));
    if (!venc_)
        load_encoder_kl();
    const VaeEncoder& e = *venc_;
    // The largest image: each level's widest channels at its size.
    uint64_t most = uint64_t(H) * W * std::max(e.in.cin, e.in.cout);
    uint32_t hh = H, ww = W;
    for (const Down& d : e.downs) {
        for (const Res& r : d.res)
            most = std::max(most, uint64_t(hh) * ww * std::max(r.c1.cin, r.c2.cout));
        if (d.down) {
            hh /= 2;
            ww /= 2;
        }
    }
    most = std::max(most, uint64_t(hh) * ww * std::max({e.mid0.c2.cout, e.out.cout, e.quant.cout}));
    if (hh != h * patch_ || ww != w * patch_)
        throw std::runtime_error("vae: not a 16x encoder for its tokens");
    MTL::Buffer* a = scratch("v_a", most * 4);
    MTL::Buffer* b = scratch("v_b", most * 4);
    MTL::Buffer* t1 = scratch("v_t1", most * 4);
    MTL::Buffer* t2 = scratch("v_t2", most * 4);
    {
        auto* in = static_cast<float*>(a->contents());
        const uint32_t c = e.in.cin;
        for (uint64_t p = 0; p < uint64_t(H) * W; ++p)
            for (uint32_t ch = 0; ch < c; ++ch)
                in[p * c + ch] = rgba[p * 4 + ch];
    }
    MTL::CommandBuffer* cb = ctx_.queue()->commandBuffer();
    cur_cb_ = cb;
    MTL::ComputeCommandEncoder* enc = encoder(cb, 0, -1);
    conv(enc, e.in, a, H, W, b);
    MTL::Buffer* x = b;
    MTL::Buffer* y = a;
    hh = H;
    ww = W;
    for (const Down& d : e.downs) {
        for (const Res& r : d.res) {
            res_block(enc, r, x, hh, ww, y, t1, t2);
            std::swap(x, y);
        }
        if (d.down) {
            conv(enc, d.conv, x, hh, ww, y);
            std::swap(x, y);
            hh /= 2;
            ww /= 2;
        }
        probe(enc, "enc_down", int(&d - e.downs.data()), x,
              uint64_t(hh) * ww * (d.down ? d.conv.cout : d.res.back().c2.cout));
    }
    res_block(enc, e.mid0, x, hh, ww, y, t1, t2);
    std::swap(x, y);
    vae_attention(enc, e.attn_g, &e.attn_b, e.qkv, e.proj, x, hh, ww, t1, t2, y);
    res_block(enc, e.mid1, x, hh, ww, y, t1, t2);
    std::swap(x, y);
    probe(enc, "enc_mid", -1, x, uint64_t(hh) * ww * e.mid1.c2.cout);
    norm_silu(enc, x, e.out_g, t1, hh * ww, e.out.cin, &e.out_b);
    conv(enc, e.out, t1, hh, ww, y);
    conv(enc, e.quant, y, hh, ww, x);
    enc->endEncoding();
    finish(cb);
    // The mean (the first z_ of each pixel's values), 2x2 patches packed into tokens and normalized.
    const auto* o = static_cast<const float*>(x->contents());
    const uint32_t oc = e.quant.cout, p = patch_;
    std::vector<float> lat(uint64_t(h) * w * lat_);
    for (uint32_t ty = 0; ty < h; ++ty)
        for (uint32_t tx = 0; tx < w; ++tx)
            for (uint32_t ch = 0; ch < z_; ++ch)
                for (uint32_t py = 0; py < p; ++py)
                    for (uint32_t px = 0; px < p; ++px) {
                        const uint32_t i = (ch * p + py) * p + px;
                        const float v = o[((uint64_t(ty) * p + py) * ww + tx * p + px) * oc + ch];
                        lat[(uint64_t(ty) * w + tx) * lat_ + i] = (v - lat_mean_[i]) / lat_std_[i];
                    }
    return lat;
}

std::vector<float> Engine::decode(std::span<const float> latents, uint32_t h, uint32_t w)
{
    if (latents.size() != uint64_t(h) * w * lat_)
        throw std::runtime_error(std::format("latents must be [h * w][{}]", lat_));
    // The VAE's own latent image: a token's 2x2 patch spread over pixels where it packs one.
    const uint32_t lh = h * patch_, lw = w * patch_;
    // The largest image: the last up block's input, at the output's size, and the widest channels at its scale.
    uint64_t most = uint64_t(lh) * lw * z_;
    uint32_t hh = lh, ww = lw;
    most = std::max(most, uint64_t(hh) * ww * conv_in_.cout);
    for (const Up& u : ups_) {
        most = std::max(most, uint64_t(hh) * ww * std::max(u.cin, u.cout));
        if (u.up) {
            hh *= 2;
            ww *= 2;
            most = std::max(most, uint64_t(hh) * ww * u.cout);
        }
    }
    const uint32_t H = hh, W = ww;
    if (H != 16 * h || W != 16 * w)
        throw std::runtime_error("vae: not a 16x decoder for its tokens");
    MTL::Buffer* a = scratch("v_a", most * 4);
    MTL::Buffer* b = scratch("v_b", most * 4);
    MTL::Buffer* t1 = scratch("v_t1", most * 4);
    MTL::Buffer* t2 = scratch("v_t2", most * 4);
    MTL::Buffer* keep = scratch("v_keep", most * 4);
    auto* lat = static_cast<float*>(a->contents());
    const uint32_t p = patch_;
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x)
            for (uint32_t ch = 0; ch < z_; ++ch)
                for (uint32_t py = 0; py < p; ++py)
                    for (uint32_t px = 0; px < p; ++px) {
                        const uint32_t i = (ch * p + py) * p + px;
                        const float v = latents[(uint64_t(y) * w + x) * lat_ + i];
                        lat[((uint64_t(y) * p + py) * lw + x * p + px) * z_ + ch] = v * lat_std_[i] + lat_mean_[i];
                    }

    MTL::CommandBuffer* cb = ctx_.queue()->commandBuffer();
    cur_cb_ = cb;
    MTL::ComputeCommandEncoder* enc = encoder(cb, 0, -1);
    probe(enc, "vae_in", -1, a, uint64_t(lh) * lw * z_);
    conv(enc, post_quant_, a, lh, lw, b);
    probe(enc, "vae_pq", -1, b, uint64_t(lh) * lw * z_);
    conv(enc, conv_in_, b, lh, lw, a);
    probe(enc, "vae_conv_in", -1, a, uint64_t(lh) * lw * conv_in_.cout);
    hh = lh;
    ww = lw;
    res_block(enc, mid0_, a, hh, ww, b, t1, t2);
    vae_attention(enc, attn_g_, kl_ ? &attn_b_ : nullptr, qkv_, attn_proj_, b, hh, ww, t1, t2, a);
    res_block(enc, mid1_, b, hh, ww, a, t1, t2);
    probe(enc, "vae_mid", -1, a, uint64_t(hh) * ww * mid1_.c2.cout);
    MTL::Buffer* x = a;
    MTL::Buffer* y = b;
    for (const Up& u : ups_) {
        const uint32_t rows = hh * ww;
        if (u.up && residual_) { // the block's input, for the duplicating shortcut
            const uint32_t n = rows * u.cin;
            dispatch(enc, "copy_f32", {x, keep}, &n, sizeof(n), MTL::Size(n, 1, 1), MTL::Size(256, 1, 1));
        }
        for (const Res& r : u.res) {
            res_block(enc, r, x, hh, ww, y, t1, t2);
            std::swap(x, y);
        }
        if (u.up) {
            const UpArgs ua{hh, ww, u.cout, u.cout, u.ft, u.cout * u.ft * 4 / u.cin};
            dispatch(enc, "upsample2x", {x, t1}, &ua, sizeof(ua), MTL::Size(u.cout, 2 * ww, 2 * hh),
                     MTL::Size(64, 1, 1));
            conv(enc, u.conv, t1, 2 * hh, 2 * ww, y);
            if (residual_) {
                const UpArgs da{hh, ww, u.cin, u.cout, u.ft, u.cout * u.ft * 4 / u.cin};
                dispatch(enc, "dup_up_add", {keep, y}, &da, sizeof(da), MTL::Size(u.cout, 2 * ww, 2 * hh),
                         MTL::Size(64, 1, 1));
            }
            std::swap(x, y);
            hh *= 2;
            ww *= 2;
        }
        probe(enc, "vae_up", int(&u - ups_.data()), x, uint64_t(hh) * ww * (u.up ? u.conv.cout : u.cout));
    }
    norm_silu(enc, x, out_g_, t1, hh * ww, conv_out_.cin, kl_ ? &out_b_ : nullptr);
    conv(enc, conv_out_, t1, hh, ww, y);
    enc->endEncoding();
    finish(cb);
    // RGB (opaque) or RGBA.
    const auto* o = static_cast<const float*>(y->contents());
    const uint32_t oc = conv_out_.cout;
    std::vector<float> img(uint64_t(H) * W * 4);
    for (uint64_t px = 0; px < uint64_t(H) * W; ++px)
        for (uint32_t ch = 0; ch < 4; ++ch)
            img[px * 4 + ch] = ch < oc ? std::clamp(o[px * oc + ch], -1.0f, 1.0f) : 1.0f;
    return img;
}

std::vector<std::pair<std::string, float>> prompt_weights(const std::string& prompt)
{
    // Escaped parentheses out of the way, as two unused bytes each.
    std::string text;
    for (size_t i = 0; i < prompt.size(); ++i)
        if (prompt[i] == '\\' && i + 1 < prompt.size() && (prompt[i + 1] == '(' || prompt[i + 1] == ')')) {
            text += prompt[i + 1] == ')' ? std::string("\0\1", 2) : std::string("\0\2", 2);
            ++i;
        } else {
            text += prompt[i];
        }
    // Top-level groups: text, and parenthesized runs (an unclosed one runs to the end, unmatched ")" stay in text).
    auto groups = [](const std::string& s) {
        std::vector<std::string> out;
        std::string cur;
        int level = 0;
        for (char c : s) {
            if (c == '(') {
                if (level == 0) {
                    if (!cur.empty())
                        out.push_back(cur);
                    cur = "(";
                } else {
                    cur += c;
                }
                ++level;
            } else if (c == ')') {
                --level;
                if (level == 0) {
                    out.push_back(cur + ")");
                    cur.clear();
                } else {
                    cur += c;
                }
            } else {
                cur += c;
            }
        }
        if (!cur.empty())
            out.push_back(cur);
        return out;
    };
    std::vector<std::pair<std::string, float>> out;
    // Nesting deeper than kMaxDepth is text (each level copies the string, and the weights compound).
    constexpr int kMaxDepth = 16;
    auto parse = [&](auto& self, const std::string& s, float weight, int depth) -> void {
        for (std::string x : groups(s)) {
            if (x.size() >= 2 && x.front() == '(' && x.back() == ')' && depth < kMaxDepth) {
                x = x.substr(1, x.size() - 2);
                float w = weight * 1.1f;
                if (const size_t colon = x.rfind(':'); colon != std::string::npos && colon > 0) {
                    std::string num = x.substr(colon + 1);
                    const size_t a = num.find_first_not_of(" \t\n\r"), b = num.find_last_not_of(" \t\n\r");
                    num = a == std::string::npos ? "" : num.substr(a, b - a + 1);
                    char* end = nullptr;
                    const double v = num.empty() ? 0 : std::strtod(num.c_str(), &end);
                    if (!num.empty() && end && *end == '\0' && std::isfinite(v)) {
                        w = float(std::clamp(v, -100.0, 100.0));
                        x = x.substr(0, colon);
                    }
                }
                self(self, x, w, depth + 1);
            } else {
                out.emplace_back(x, weight);
            }
        }
    };
    parse(parse, text, 1.0f, 0);
    for (auto& [s, w] : out) {
        std::string u;
        for (size_t i = 0; i < s.size(); ++i)
            if (s[i] == '\0' && i + 1 < s.size() && (s[i + 1] == '\1' || s[i + 1] == '\2')) {
                u += s[i + 1] == '\1' ? ')' : '(';
                ++i;
            } else {
                u += s[i];
            }
        s = u;
    }
    return out;
}

Lora parse_lora(const std::string& spec)
{
    Lora l;
    l.path = spec;
    for (const auto& [suffix, merge] : {std::pair{":merge", true}, {":unmerged", false}})
        if (l.path.ends_with(suffix)) {
            l.path.resize(l.path.size() - std::strlen(suffix));
            l.merge = merge;
            break;
        }
    const size_t colon = l.path.rfind(':');
    if (colon != std::string::npos && colon + 1 < l.path.size()) {
        char* end = nullptr;
        const float v = std::strtof(l.path.c_str() + colon + 1, &end);
        if (end && *end == '\0') {
            l.scale = v;
            l.path.resize(colon);
        }
    }
    if (l.path.empty())
        throw std::invalid_argument("names no LoRA file");
    return l;
}

} // namespace mako::image
