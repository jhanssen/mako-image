#include "image/engine.hpp"

#include "image/anima.hpp"
#include "image/er_sde.hpp"

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

struct HeadsArgs {
    uint32_t rows, ld, col0, dim, n_pad, row0, rope;
};
struct HeadsNormArgs {
    uint32_t ld, col0, dim, n_pad, row0, rope;
    float eps;
};
struct AttnArgs {
    uint32_t n_q, q_pad, n_k, k_pad, heads, kv_heads, q_pos0, block, width;
    float scale;
};
struct UpArgs {
    uint32_t h, w, c, c_out, ft, repeats;
};
struct TransposeArgs {
    uint32_t rows, cols, ldx, col0;
};
struct LoraArgs {
    uint32_t k, rows, rank;
    float scale;
};

uint32_t round_up(uint32_t x, uint32_t m) { return (x + m - 1) / m * m; }

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
uint16_t to_bf16(float f)
{
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return uint16_t((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

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
    : ConvRuntime(ctx, opts.telemetry), text_files_(open_files(parts.text_encoder)), vae_files_(open_files(parts.vae))
{
    load_text(parts.text_encoder);
    if (parts.text_theta > 0)
        tTheta_ = parts.text_theta;

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

    auto t = text::Tokenizer::load_json(parts.tokenizer);
    if (!t)
        throw std::runtime_error(t.error());
    tok_ = std::move(*t);
}

Engine::Engine(mtl::Context& ctx, const Options& opts) : ConvRuntime(ctx, opts.telemetry) {}

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

void Engine::load_text(const std::string& tdir)
{
    // eps, the rotary base and the model type from the encoder's config, where it has one beside the weights.
    double rot_share = 1.0;
    if (std::filesystem::exists(tdir + "/config.json")) {
        auto j = read_json_file(tdir + "/config.json");
        const auto& t = j.contains("text_config") ? j["text_config"] : j;
        if (t.contains("rms_norm_eps") && t["rms_norm_eps"].is_number())
            tEps_ = float(t["rms_norm_eps"].get_number());
        if (t.contains("rope_theta") && t["rope_theta"].is_number())
            tTheta_ = float(t["rope_theta"].get_number());
        const bool rp = t.contains("rope_parameters");
        if (rp && t["rope_parameters"].contains("rope_theta"))
            tTheta_ = float(t["rope_parameters"]["rope_theta"].get_number());
        qwen35_ = t.contains("model_type") && t["model_type"].is_string() &&
                  t["model_type"].get_string().starts_with("qwen3_5");
        if (qwen35_ && rp && t["rope_parameters"].contains("partial_rotary_factor"))
            rot_share = t["rope_parameters"]["partial_rotary_factor"].get_number();
    }
    // The language model's tensors, under the full model's prefix or the bare one's.
    std::string p;
    for (const char* prefix : {"model.language_model.", "language_model.", "model."})
        if (text_files_.find(std::string(prefix) + "embed_tokens.weight")) {
            p = prefix;
            break;
        }
    embd_ = text_files_.find(p + "embed_tokens.weight");
    if (!embd_ || embd_.tensor->type != gguf::Type::BF16)
        throw std::runtime_error("text encoder: embed_tokens missing or not BF16");
    tE_ = uint32_t(embd_.tensor->dims[0]);
    // Qwen3.5 stores its norms' scales zero-centred.
    auto norm_w = [&](const std::string& name) {
        return qwen35_ ? upload_plus_one(text_files_, name) : upload(text_files_, name, false);
    };
    for (uint32_t i = 0;; ++i) {
        const std::string l = std::format("{}layers.{}.", p, i);
        if (!text_files_.find(l + "input_layernorm.weight"))
            break;
        TextLayer t;
        t.attn_norm = norm_w(l + "input_layernorm.weight");
        if (text_files_.find(l + "linear_attn.in_proj_qkv.weight")) {
            const std::string a = l + "linear_attn.";
            t.linear = true;
            t.q = upload(text_files_, a + "in_proj_qkv.weight", true);
            t.z = upload(text_files_, a + "in_proj_z.weight", true);
            const std::string ba[2] = {a + "in_proj_b.weight", a + "in_proj_a.weight"};
            t.ba = upload_rows(text_files_, ba);
            t.conv = upload(text_files_, a + "conv1d.weight", false);
            t.a_log = upload(text_files_, a + "A_log", false);
            t.dt_bias = upload(text_files_, a + "dt_bias", false);
            t.q_norm = upload(text_files_, a + "norm.weight", false);
            t.o = upload(text_files_, a + "out_proj.weight", true);
        } else {
            t.q = upload(text_files_, l + "self_attn.q_proj.weight", true);
            t.k = upload(text_files_, l + "self_attn.k_proj.weight", true);
            t.v = upload(text_files_, l + "self_attn.v_proj.weight", true);
            t.o = upload(text_files_, l + "self_attn.o_proj.weight", true);
            t.q_norm = norm_w(l + "self_attn.q_norm.weight");
            t.k_norm = norm_w(l + "self_attn.k_norm.weight");
        }
        t.ffn_norm = norm_w(l + "post_attention_layernorm.weight");
        t.gate = upload(text_files_, l + "mlp.gate_proj.weight", true);
        t.up = upload(text_files_, l + "mlp.up_proj.weight", true);
        t.down = upload(text_files_, l + "mlp.down_proj.weight", true);
        text_.push_back(std::move(t));
    }
    if (text_.empty())
        throw std::runtime_error("text encoder: no layers");
    if (text_files_.find(p + "norm.weight"))
        text_norm_ = norm_w(p + "norm.weight");
    const auto attn = std::ranges::find_if(text_, [](const TextLayer& t) { return !t.linear; });
    if (attn == text_.end())
        throw std::runtime_error("text encoder: no attention layer");
    tHead_ = uint32_t(attn->q_norm.dims[0]);
    if (tHead_ != kHeadDim && !(qwen35_ && tHead_ == 256))
        throw std::runtime_error(std::format("text encoder: {}-wide heads", tHead_));
    // Qwen3.5's queries carry an output gate per head: [heads][query, gate].
    tHeads_ = uint32_t(attn->q.dims[1] / (tHead_ * (qwen35_ ? 2 : 1)));
    tKv_ = uint32_t(attn->k.dims[1] / tHead_);
    tRot_ = uint32_t(tHead_ * rot_share);
    for (const TextLayer& t : text_)
        if (t.linear) {
            tValueHeads_ = uint32_t(t.a_log.dims[0]);
            tKeyHeads_ = uint32_t((t.q.dims[1] - uint64_t(tValueHeads_) * 128) / 256);
            if (t.q_norm.dims[0] != 128 || t.z.dims[1] != uint64_t(tValueHeads_) * 128 || !tKeyHeads_ ||
                tValueHeads_ % tKeyHeads_ || t.conv.dims[0] > 8)
                throw std::runtime_error("text encoder: delta rule heads not 128 wide");
            break;
        }
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
    const uint32_t h = r.height / 16, w = r.width / 16, N = h * w;
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
        return decode(x0, h, w);
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
    return decode(x, h, w);
}

void Engine::set_control(std::vector<float>, std::vector<float>)
{
    throw std::runtime_error(std::format("{}: takes no control adapter", name()));
}

void Engine::control_on(bool, bool) {}

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

Engine::Tensor Engine::upload_rows(const safetensors::Files& files, std::span<const std::string> names)
{
    uint64_t rows = 0, k = 0;
    bool f16 = true;
    for (const auto& n : names) {
        const auto r = files.find(n);
        if (!r ||
            (r.tensor->type != gguf::Type::BF16 && r.tensor->type != gguf::Type::F32 &&
             r.tensor->type != gguf::Type::F16) ||
            r.tensor->dims.size() != 2 || (k && r.tensor->dims[0] != k))
            throw std::runtime_error(n + ": missing, or not a BF16, F16 or F32 matrix of the others' width");
        k = r.tensor->dims[0];
        rows += r.tensor->dims[1];
        f16 = f16 && r.tensor->type == gguf::Type::F16;
    }
    Tensor t = reserve(rows * k * 2, f16 ? gguf::Type::F16 : gguf::Type::BF16, {k, rows});
    char* dst = static_cast<char*>(t.buf->contents()) + t.off;
    for (const auto& n : names) {
        const auto r = files.find(n);
        if (r.tensor->type == gguf::Type::BF16 || f16) {
            gguf::read_exact(r.file->fd(), dst, r.tensor->bytes, r.tensor->offset);
            dst += r.tensor->bytes;
        } else {
            const auto f = read_floats(*r.file, *r.tensor);
            auto* o = reinterpret_cast<uint16_t*>(dst);
            for (size_t i = 0; i < f.size(); ++i)
                o[i] = to_bf16(f[i]);
            dst += f.size() * 2;
        }
    }
    return t;
}

Engine::Tensor Engine::upload(const safetensors::Files& files, const std::string& name, bool matrix)
{
    const auto r = files.find(name);
    if (!r)
        throw std::runtime_error("missing tensor " + name);
    if (matrix && r.tensor->type == gguf::Type::F32 && r.tensor->dims.size() == 2) {
        const std::string n[1] = {name};
        return upload_rows(files, n);
    }
    return upload_raw(r.file->fd(), r.tensor->offset, r.tensor->bytes, r.tensor->type, r.tensor->dims, name, matrix);
}

Engine::Tensor Engine::upload_f32(const safetensors::Files& files, const std::string& name)
{
    const auto f = floats(files, name);
    Tensor t = reserve(f.size() * 4, gguf::Type::F32, {f.size()});
    std::memcpy(static_cast<char*>(t.buf->contents()) + t.off, f.data(), f.size() * 4);
    return t;
}

Engine::Tensor Engine::upload_plus_one(const safetensors::Files& files, const std::string& name)
{
    const auto r = files.find(name);
    if (!r)
        throw std::runtime_error("missing tensor " + name);
    const auto f = read_floats(*r.file, *r.tensor);
    Tensor t = reserve(f.size() * 4, gguf::Type::F32, {f.size()});
    auto* o = reinterpret_cast<float*>(static_cast<char*>(t.buf->contents()) + t.off);
    for (size_t i = 0; i < f.size(); ++i)
        o[i] = f[i] + 1.0f;
    return t;
}

void Engine::embed(std::span<const int32_t> ids, float* out) const
{
    const uint64_t vocab = embd_.tensor->dims[1], row = uint64_t(tE_) * 2;
    std::vector<uint16_t> raw(tE_);
    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] < 0 || uint64_t(ids[i]) >= vocab)
            throw std::runtime_error("token out of range");
        gguf::read_exact(embd_.file->fd(), raw.data(), row, embd_.tensor->offset + uint64_t(ids[i]) * row);
        auto* o = reinterpret_cast<uint32_t*>(out + i * tE_);
        for (uint32_t j = 0; j < tE_; ++j)
            o[j] = uint32_t(raw[j]) << 16;
    }
}

void Engine::add(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* x, MTL::Buffer* z, MTL::Buffer* y, uint32_t n)
{
    dispatch(enc, "add", {x, z, y}, &n, sizeof(n), MTL::Size(n, 1, 1), MTL::Size(256, 1, 1));
}

void Engine::unary(MTL::ComputeCommandEncoder*& enc, const char* fn, MTL::Buffer* x, uint32_t n)
{
    dispatch(enc, fn, {x}, &n, sizeof(n), MTL::Size(n, 1, 1), MTL::Size(256, 1, 1));
}

void Engine::heads(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* x, uint32_t ld, uint32_t col0, uint32_t n_heads,
                   uint32_t T, uint32_t n_pad, MTL::Buffer* cs, uint32_t rope, MTL::Buffer* y)
{
    const HeadsArgs a{T, ld, col0, kHeadDim, n_pad, 0, rope};
    dispatch(enc, "heads_half", {x, cs ? cs : x, y}, &a, sizeof(a), MTL::Size(kHeadDim / 2, T, n_heads),
             MTL::Size(64, 1, 1));
    if (n_pad > T) {
        const uint32_t c[4] = {kHeadDim, n_pad, T, 0};
        dispatch(enc, "heads_clear", {y}, c, sizeof(c), MTL::Size(kHeadDim, n_pad - T, n_heads), MTL::Size(128, 1, 1));
    }
}

void Engine::heads_norm(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* x, uint32_t ld, uint32_t col0, uint32_t n_heads,
                        uint32_t T, uint32_t n_pad, MTL::Buffer* cs, uint64_t cs_off, uint32_t rope, const Tensor& w,
                        float eps, uint32_t row0, MTL::Buffer* y)
{
    const HeadsNormArgs a{ld, col0, kHeadDim, n_pad, row0, rope, eps};
    enc->setComputePipelineState(pso("heads_norm_half"));
    enc->setBuffer(x, 0, 0);
    enc->setBuffer(cs ? cs : x, cs ? cs_off : 0, 1);
    enc->setBuffer(y, 0, 2);
    enc->setBuffer(w.buf, w.off, 3);
    enc->setBytes(&a, sizeof(a), 4);
    enc->dispatchThreadgroups(MTL::Size(T, n_heads, 1), MTL::Size(kHeadDim / 2, 1, 1));
    after(enc, "heads_norm_half");
}

void Engine::attention(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* q, MTL::Buffer* k, MTL::Buffer* v,
                       MTL::Buffer* o, uint32_t n_q, uint32_t q_pad, uint32_t n_k, uint32_t k_pad, uint32_t n_heads,
                       uint32_t kv_heads, uint32_t q_pos0, uint32_t block)
{
    const AttnArgs a{n_q, q_pad, n_k, k_pad, n_heads, kv_heads, q_pos0, block, n_heads * kHeadDim,
                     1.0f / std::sqrt(float(kHeadDim))};
    dispatch(enc, "flash_attn_sg_128_16x32_4", {q, k, v, o}, &a, sizeof(a),
             MTL::Size(q_pad / kAttnRows, n_heads, 1), MTL::Size(128, 1, 1), false);
}

std::vector<float> Engine::encode_text(std::span<const int32_t> ids, uint32_t drop, std::span<const uint32_t> taps,
                                       bool normed)
{
    const uint32_t T = uint32_t(ids.size()), E = tE_, kv = tKv_ * kHeadDim, qw = tHeads_ * kHeadDim;
    if (T <= drop)
        throw std::runtime_error("the prompt is all template");
    uint32_t layers = uint32_t(text_.size());
    if (!taps.empty()) {
        layers = *std::max_element(taps.begin(), taps.end());
        if (layers > text_.size())
            throw std::runtime_error("text encoder: a tapped layer past its last");
    }
    const uint32_t nt = taps.empty() ? 1 : uint32_t(taps.size());
    const uint32_t q_pad = round_up(T, kAttnRows), k_pad = round_up(T, 128);
    MTL::Buffer* x = scratch("t_x", uint64_t(T) * E * 4);
    MTL::Buffer* xn = scratch("t_xn", uint64_t(T) * E * 4);
    MTL::Buffer* q = scratch("t_q", uint64_t(T) * qw * 4);
    MTL::Buffer* k = scratch("t_k", uint64_t(T) * kv * 4);
    MTL::Buffer* v = scratch("t_v", uint64_t(T) * kv * 4);
    MTL::Buffer* qh = scratch("t_qh", uint64_t(tHeads_) * q_pad * kHeadDim * 2);
    MTL::Buffer* kh = scratch("t_kh", uint64_t(tKv_) * k_pad * kHeadDim * 2);
    MTL::Buffer* vh = scratch("t_vh", uint64_t(tKv_) * k_pad * kHeadDim * 2);
    MTL::Buffer* att = scratch("t_att", uint64_t(T) * qw * 4);
    MTL::Buffer* y = scratch("t_y", uint64_t(T) * E * 4);
    const uint32_t pairs = tRot_ / 2;
    MTL::Buffer* cs = scratch("t_cs", uint64_t(T) * pairs * 8);
    MTL::Buffer* out = scratch("t_out", uint64_t(T) * nt * E * 4);
    embed(ids, static_cast<float*>(x->contents()));
    // Text positions are the same on the three rotary axes, which makes the multi-axis rotation the plain one.
    auto* c = static_cast<float*>(cs->contents());
    for (uint32_t t = 0; t < T; ++t)
        for (uint32_t i = 0; i < pairs; ++i) {
            const float inv = 1.0f / std::pow(tTheta_, float(2 * i) / float(tRot_));
            const float ang = float(t) * inv;
            c[(uint64_t(t) * pairs + i) * 2] = std::cos(ang);
            c[(uint64_t(t) * pairs + i) * 2 + 1] = std::sin(ang);
        }

    MTL::CommandBuffer* cb = ctx_.queue()->commandBuffer();
    cur_cb_ = cb;
    MTL::ComputeCommandEncoder* enc = encoder(cb, 0, -1);
    // The first token's outlier channels grow to thousands and cancel in the last layers: activations in bfloat
    // alone leave its final state a few percent off.
    bf16_split_ = true;
    // Hidden state i (0: the embeddings) into its column of the taps' rows.
    auto tap = [&](uint32_t i) {
        for (uint32_t j = 0; j < taps.size(); ++j)
            if (taps[j] == i) {
                const uint32_t a[2] = {E, nt * E};
                enc->setComputePipelineState(pso("copy_rows"));
                enc->setBuffer(x, 0, 0);
                enc->setBuffer(out, uint64_t(j) * E * 4, 1);
                enc->setBytes(a, sizeof(a), 2);
                enc->dispatchThreads(MTL::Size(E, T, 1), MTL::Size(256, 1, 1));
                after(enc, "copy_rows");
            }
    };
    tap(0);
    for (uint32_t il = 0; il < layers; ++il) {
        const TextLayer& L = text_[il];
        cur_layer_ = il;
        norm(enc, x, &L.attn_norm, xn, T, E, E, tEps_);
        if (qwen35_) {
            if (L.linear)
                text_linear(enc, L, xn, y, T);
            else
                text_gated_attention(enc, L, xn, y, T, cs);
            add(enc, x, y, x, T * E);
            norm(enc, x, &L.ffn_norm, xn, T, E, E, tEps_);
            ffn_dense(enc, L.gate, L.up, L.down, xn, y, 0.0f, T);
            add(enc, x, y, x, T * E);
            tap(il + 1);
            continue;
        }
        proj(enc, L.q, xn, 0, q, 0, T);
        proj(enc, L.k, xn, 0, k, 0, T);
        proj(enc, L.v, xn, 0, v, 0, T);
        norm(enc, q, &L.q_norm, q, T * tHeads_, kHeadDim, kHeadDim, tEps_);
        norm(enc, k, &L.k_norm, k, T * tKv_, kHeadDim, kHeadDim, tEps_);
        heads(enc, q, qw, 0, tHeads_, T, q_pad, cs, 1, qh);
        heads(enc, k, kv, 0, tKv_, T, k_pad, cs, 1, kh);
        heads(enc, v, kv, 0, tKv_, T, k_pad, nullptr, 0, vh);
        attention(enc, qh, kh, vh, att, T, q_pad, T, k_pad, tHeads_, tKv_, 0, UINT32_MAX);
        proj(enc, L.o, att, 0, y, 0, T);
        add(enc, x, y, x, T * E);
        norm(enc, x, &L.ffn_norm, xn, T, E, E, tEps_);
        ffn_dense(enc, L.gate, L.up, L.down, xn, y, 0.0f, T);
        add(enc, x, y, x, T * E);
        tap(il + 1);
    }
    if (normed) {
        if (!text_norm_.buf)
            throw std::runtime_error("text encoder: no final norm");
        norm(enc, x, &text_norm_, x, T, E, E, tEps_);
    }
    bf16_split_ = false;
    enc->endEncoding();
    finish(cb);
    const auto* src = static_cast<const float*>((taps.empty() ? x : out)->contents());
    return std::vector<float>(src + uint64_t(drop) * nt * E, src + uint64_t(T) * nt * E);
}

void Engine::text_linear(MTL::ComputeCommandEncoder*& enc, const TextLayer& L, MTL::Buffer* xn, MTL::Buffer* y,
                         uint32_t T)
{
    const uint32_t KH = tKeyHeads_, VH = tValueHeads_, width = (2 * KH + VH) * 128, vw = VH * 128;
    MTL::Buffer* m = scratch("g_m", uint64_t(T) * width * 4);
    MTL::Buffer* mc = scratch("g_mc", uint64_t(T) * width * 4);
    MTL::Buffer* z = scratch("g_z", uint64_t(T) * vw * 4);
    MTL::Buffer* ba = scratch("g_ba", uint64_t(T) * 2 * VH * 4);
    MTL::Buffer* q = scratch("g_q", uint64_t(T) * vw * 4);
    MTL::Buffer* k = scratch("g_k", uint64_t(T) * vw * 4);
    MTL::Buffer* v = scratch("g_v", uint64_t(T) * vw * 4);
    MTL::Buffer* g = scratch("g_g", uint64_t(T) * vw * 4);
    MTL::Buffer* beta = scratch("g_beta", uint64_t(T) * VH * 4);
    MTL::Buffer* o = scratch("g_o", uint64_t(T) * vw * 4);
    // Each layer's own state, zeroed now: the GPU runs the layers after all are encoded.
    MTL::Buffer* state = scratch(std::format("g_state{}", cur_layer_), uint64_t(VH) * 128 * 128 * 4);
    std::memset(state->contents(), 0, state->length());
    proj(enc, L.q, xn, 0, m, 0, T);
    proj(enc, L.z, xn, 0, z, 0, T);
    proj(enc, L.ba, xn, 0, ba, 0, T);
    const uint32_t ca[3] = {width, uint32_t(L.conv.dims[0]), T};
    enc->setComputePipelineState(pso("gdn_conv"));
    enc->setBuffer(m, 0, 0);
    enc->setBuffer(L.conv.buf, L.conv.off, 1);
    enc->setBuffer(mc, 0, 2);
    enc->setBytes(ca, sizeof(ca), 3);
    enc->dispatchThreads(MTL::Size(width, 1, 1), MTL::Size(256, 1, 1));
    after(enc, "gdn_conv");
    struct {
        uint32_t k_heads, v_heads, conv_width;
        float eps;
    } pa{KH, VH, width, 1e-6f};
    enc->setComputePipelineState(pso("gdn_prep"));
    enc->setBuffer(mc, 0, 0);
    enc->setBuffer(ba, 0, 1);
    enc->setBuffer(L.a_log.buf, L.a_log.off, 2);
    enc->setBuffer(L.dt_bias.buf, L.dt_bias.off, 3);
    for (uint32_t i = 0; i < 5; ++i)
        enc->setBuffer(std::array{q, k, v, g, beta}[i], 0, 4 + i);
    enc->setBytes(&pa, sizeof(pa), 9);
    enc->dispatchThreadgroups(MTL::Size(VH, T, 1), MTL::Size(128, 1, 1));
    after(enc, "gdn_prep");
    // The delta rule with a decay per head: the per-channel scan with each head's decay on all of its channels.
    const uint32_t sa[6] = {128, T, vw, vw, VH, 0};
    enc->setComputePipelineState(pso("kda_scan"));
    for (uint32_t i = 0; i < 7; ++i)
        enc->setBuffer(std::array{q, k, v, g, beta, state, o}[i], 0, i);
    enc->setBytes(sa, sizeof(sa), 7);
    enc->setBuffer(state, 0, 8);
    enc->dispatchThreadgroups(MTL::Size(VH, 4, 1), MTL::Size(128, 1, 1));
    after(enc, "kda_scan");
    const struct {
        uint32_t heads;
        float eps;
    } ga{VH, tEps_};
    enc->setComputePipelineState(pso("gated_rms"));
    enc->setBuffer(o, 0, 0);
    enc->setBuffer(z, 0, 1);
    enc->setBuffer(L.q_norm.buf, L.q_norm.off, 2);
    enc->setBytes(&ga, sizeof(ga), 3);
    enc->dispatchThreadgroups(MTL::Size(VH, T, 1), MTL::Size(128, 1, 1));
    after(enc, "gated_rms");
    proj(enc, L.o, o, 0, y, 0, T);
}

void Engine::text_gated_attention(MTL::ComputeCommandEncoder*& enc, const TextLayer& L, MTL::Buffer* xn,
                                  MTL::Buffer* y, uint32_t T, MTL::Buffer* cs)
{
    const uint32_t D = tHead_, H = tHeads_, KV = tKv_, qw = 2 * H * D, kw = KV * D;
    const uint32_t q_pad = round_up(T, kAttnRows), k_pad = round_up(T, 128);
    MTL::Buffer* q = scratch("a_q", uint64_t(T) * qw * 4);
    MTL::Buffer* k = scratch("a_k", uint64_t(T) * kw * 4);
    MTL::Buffer* v = scratch("a_v", uint64_t(T) * kw * 4);
    MTL::Buffer* qh = scratch("a_qh", uint64_t(H) * q_pad * D * 2);
    MTL::Buffer* kh = scratch("a_kh", uint64_t(KV) * k_pad * D * 2);
    MTL::Buffer* vh = scratch("a_vh", uint64_t(KV) * k_pad * D * 2);
    MTL::Buffer* att = scratch("a_att", uint64_t(T) * H * D * 4);
    std::memset(qh->contents(), 0, qh->length());
    std::memset(kh->contents(), 0, kh->length());
    std::memset(vh->contents(), 0, vh->length());
    proj(enc, L.q, xn, 0, q, 0, T);
    proj(enc, L.k, xn, 0, k, 0, T);
    proj(enc, L.v, xn, 0, v, 0, T);
    struct HeadsPartialArgs {
        uint32_t ld, col0, stride, dim, rot, n_pad;
        float eps;
    };
    auto heads = [&](MTL::Buffer* x, const HeadsPartialArgs& a, const Tensor* w, uint32_t n, MTL::Buffer* out) {
        enc->setComputePipelineState(pso("heads_partial"));
        enc->setBuffer(x, 0, 0);
        enc->setBuffer(cs, 0, 1);
        enc->setBuffer(out, 0, 2);
        enc->setBuffer(w ? w->buf : x, w ? w->off : 0, 3);
        enc->setBytes(&a, sizeof(a), 4);
        enc->dispatchThreadgroups(MTL::Size(T, n, 1), MTL::Size(D, 1, 1));
        after(enc, "heads_partial");
    };
    heads(q, {qw, 0, 2 * D, D, tRot_, q_pad, tEps_}, &L.q_norm, H, qh);
    heads(k, {kw, 0, D, D, tRot_, k_pad, tEps_}, &L.k_norm, KV, kh);
    heads(v, {kw, 0, D, D, 0, k_pad, 0.0f}, nullptr, KV, vh);
    const AttnArgs a{T, q_pad, T, k_pad, H, KV, 0, UINT32_MAX, H * D, 1.0f / std::sqrt(float(D))};
    dispatch(enc, std::format("flash_attn_sg_{}_16x32_4", D), {qh, kh, vh, att}, &a, sizeof(a),
             MTL::Size(q_pad / kAttnRows, H, 1), MTL::Size(128, 1, 1), false);
    const uint32_t ga[5] = {H, D, qw, D, 2 * D};
    dispatch(enc, "gate_heads", {att, q}, ga, sizeof(ga), MTL::Size(D, H, T), MTL::Size(64, 1, 1));
    proj(enc, L.o, att, 0, y, 0, T);
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

void Engine::group_norm(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* x, const Tensor& g, const Tensor& b,
                        MTL::Buffer* y, uint32_t rows, uint32_t c, bool silu, float eps)
{
    struct {
        uint32_t rows, c, groups, chunk;
        float eps;
        uint32_t silu;
    } a{rows, c, 32, 256, eps, silu ? 1u : 0u};
    const uint32_t chunks = (rows + a.chunk - 1) / a.chunk;
    MTL::Buffer* part = scratch("gn_part", uint64_t(chunks) * c * 8);
    MTL::Buffer* stats = scratch("gn_stats", 32 * 8);
    dispatch(enc, "gn_partial", {x, part}, &a, sizeof(a), MTL::Size(c, chunks, 1), MTL::Size(std::min(c, 256u), 1, 1));
    dispatch(enc, "gn_stats", {part, stats}, &a, sizeof(a), MTL::Size(32, 1, 1), MTL::Size(32, 1, 1));
    enc->setComputePipelineState(pso("gn_apply"));
    enc->setBuffer(x, 0, 0);
    enc->setBuffer(stats, 0, 1);
    enc->setBuffer(g.buf, g.off, 2);
    enc->setBuffer(b.buf, b.off, 3);
    enc->setBuffer(y, 0, 4);
    enc->setBytes(&a, sizeof(a), 5);
    enc->dispatchThreads(MTL::Size(c, rows, 1), MTL::Size(std::min(c, 256u), 1, 1));
    after(enc, "gn_apply");
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

uint32_t Engine::add_lora(const std::string& path, float scale, std::optional<bool> merge)
{
    if (merge == false && !keeps_loras_unmerged())
        throw std::runtime_error(std::format("{}: {} keeps no LoRA unmerged", path, name()));
    auto file = safetensors::File::open(path);
    if (!file)
        throw std::runtime_error(file.error());
    if (const auto it = file->metadata().find("hayate_control");
        !control_lora_ && it != file->metadata().end() && it->second == "1")
        throw std::runtime_error(std::format("{}: a control adapter, not a LoRA (the model directory's \"control\")",
                                             path));
    // The adapter's alpha for every matrix, from a PEFT config in the metadata.
    std::optional<double> alpha;
    if (const auto it = file->metadata().find("lora_adapter_metadata"); it != file->metadata().end()) {
        glz::generic j;
        if (!glz::read_json(j, it->second) && j.is_object())
            for (const char* k : {"transformer.lora_alpha", "lora_alpha"})
                if (j.contains(k) && j[k].is_number())
                    alpha = j[k].get_number();
    }
    struct Pair {
        const gguf::Tensor *a = nullptr, *b = nullptr, *alpha = nullptr;
    };
    std::map<std::string, Pair> mods;
    std::map<std::string, const gguf::Tensor*> bias_diffs, weight_diffs;
    std::vector<std::string> unknown;
    static constexpr std::pair<const char*, int> kSuffixes[] = {
        {".lora.down.weight", 0}, {".lora.up.weight", 1},  {".lora_down.weight", 0},       {".lora_up.weight", 1},
        {".lora_A.weight", 0},    {".lora_B.weight", 1},   {".lora_A.default.weight", 0}, {".lora_B.default.weight", 1},
        {".alpha", 2}};
    for (const auto& t : file->tensors()) {
        std::string_view name = t.name;
        for (const char* p : {"base_model.model.", "model.diffusion_model.", "diffusion_model.", "transformer.",
                              "lora_unet_"})
            if (name.starts_with(p))
                name.remove_prefix(std::strlen(p));
        if (control_lora_ && name.starts_with("video_embedder."))
            continue; // the control adapter's input projection, which the model reads itself
        if (name.ends_with(".diff_b")) {
            bias_diffs[std::string(name.substr(0, name.size() - 7))] = &t;
            continue;
        }
        if (name.ends_with(".diff")) { // a whole weight's difference
            weight_diffs[std::string(name.substr(0, name.size() - 5))] = &t;
            continue;
        }
        bool known = false;
        for (const auto& [suffix, role] : kSuffixes)
            if (name.ends_with(suffix)) {
                Pair& p = mods[std::string(name.substr(0, name.size() - std::strlen(suffix)))];
                (role == 0 ? p.a : role == 1 ? p.b : p.alpha) = &t;
                known = true;
                break;
            }
        if (!known)
            unknown.push_back(t.name);
    }
    // Kohya's names: the module path with underscores for its dots; which underscores were dots, tried until one names
    // a weight.
    auto dotted = [&](const std::string& m, auto&& found) -> std::string {
        if (m.find('.') != std::string::npos || found(m))
            return m;
        std::vector<size_t> at;
        for (size_t i = 0; i < m.size(); ++i)
            if (m[i] == '_')
                at.push_back(i);
        if (at.size() > 16)
            return m;
        for (uint32_t bits = 1; bits < (1u << at.size()); ++bits) {
            std::string c = m;
            for (size_t i = 0; i < at.size(); ++i)
                if (bits >> i & 1)
                    c[at[i]] = '.';
            if (found(c))
                return c;
        }
        return m;
    };
    {
        std::map<std::string, Pair> renamed;
        for (auto& [m, p] : mods)
            renamed[dotted(m, [&](const std::string& c) {
                uint32_t r = 0, k = 0;
                return lora_target(c).w != nullptr || lora_cpu(c, r, k) != nullptr;
            })] = p;
        mods = std::move(renamed);
        std::map<std::string, const gguf::Tensor*> rb;
        for (auto& [m, t] : bias_diffs)
            rb[dotted(m, [&](const std::string& c) { return lora_bias(c) != nullptr; })] = t;
        bias_diffs = std::move(rb);
        std::map<std::string, const gguf::Tensor*> rw;
        for (auto& [m, t] : weight_diffs)
            rw[dotted(m, [&](const std::string& c) {
                uint32_t r = 0, k = 0;
                return lora_cpu(c, r, k) != nullptr;
            })] = t;
        weight_diffs = std::move(rw);
    }
    for (const auto& [m, p] : mods) {
        uint32_t r = 0, k = 0;
        if (!p.a || !p.b || (!lora_target(m).w && !lora_cpu(m, r, k)))
            unknown.push_back(m);
    }
    for (const auto& [m, t] : bias_diffs) {
        const Tensor* b = lora_bias(m);
        if (!b || b->type != gguf::Type::F32 || t->elements() != b->dims[0])
            unknown.push_back(m + ".diff_b");
    }
    for (const auto& [m, t] : weight_diffs) {
        uint32_t r = 0, k = 0;
        if (!lora_cpu(m, r, k) || t->elements() != uint64_t(r) * k)
            unknown.push_back(m + ".diff");
    }
    if (!unknown.empty())
        throw std::runtime_error(std::format("{}: {} keys {} does not adapt, e.g. {}", path, unknown.size(), name(),
                                             unknown.front()));
    uint32_t done = 0, rank_seen = 0, side = 0;
    for (const auto& [m, p] : mods) {
        const LoraTarget tg = lora_target(m);
        if (!tg.w) { // a small matrix on the CPU: W += s * B A there
            uint32_t rows = 0, k = 0;
            float* w = lora_cpu(m, rows, k);
            const uint32_t rank = uint32_t(p.a->dims[1]);
            if (p.a->dims[0] != k || p.b->dims[0] != rank || p.b->dims[1] != rows)
                throw std::runtime_error(std::format("{}: {} does not fit the weight it adapts", path, m));
            const double a = p.alpha ? read_floats(*file, *p.alpha).front() : alpha.value_or(double(rank));
            const auto av = read_floats(*file, *p.a), bv = read_floats(*file, *p.b);
            for (uint32_t i = 0; i < rows; ++i)
                for (uint32_t j = 0; j < k; ++j) {
                    double sum = 0;
                    for (uint32_t r = 0; r < rank; ++r)
                        sum += double(bv[uint64_t(i) * rank + r]) * av[uint64_t(r) * k + j];
                    w[uint64_t(i) * k + j] += float(double(scale) * a / double(rank) * sum);
                }
            ++done;
            continue;
        }
        const uint32_t k = uint32_t(tg.w->dims[0]), rank = uint32_t(p.a->dims[1]);
        // A is [rank][k] and B [rows][rank] (row-major), so their contiguous dimensions come first here.
        if (p.a->dims[0] != k || p.b->dims[0] != rank || p.b->dims[1] != tg.rows ||
            (tg.w->type != gguf::Type::BF16 && tg.w->type != gguf::Type::F16))
            throw std::runtime_error(std::format("{}: {} does not fit the weight it adapts", path, m));
        double a = alpha.value_or(double(rank));
        if (p.alpha)
            a = read_floats(*file, *p.alpha).front();
        const auto av = read_floats(*file, *p.a), bv = read_floats(*file, *p.b);
        if (merge != true && lora_side(m, av, bv, rank, float(double(scale) * a / double(rank)))) {
            ++done;
            ++side;
            rank_seen = rank;
            continue;
        }
        MTL::Buffer* ab = scratch("l_a", av.size() * 4);
        MTL::Buffer* bb = scratch("l_b", bv.size() * 4);
        std::memcpy(ab->contents(), av.data(), av.size() * 4);
        std::memcpy(bb->contents(), bv.data(), bv.size() * 4);
        const LoraArgs la{k, tg.rows, rank, float(double(scale) * a / double(rank))};
        MTL::CommandBuffer* cb = ctx_.queue()->commandBuffer();
        cur_cb_ = cb;
        MTL::ComputeCommandEncoder* enc = encoder(cb, 0, -1);
        enc->setComputePipelineState(pso(tg.w->type == gguf::Type::F16 ? "lora_merge_f16" : "lora_merge"));
        enc->setBuffer(tg.w->buf, tg.w->off + uint64_t(tg.row0) * k * 2, 0);
        enc->setBuffer(ab, 0, 1);
        enc->setBuffer(bb, 0, 2);
        enc->setBytes(&la, sizeof(la), 3);
        enc->dispatchThreads(MTL::Size(k, tg.rows, 1), MTL::Size(256, 1, 1));
        enc->endEncoding();
        finish(cb);
        ++done;
        rank_seen = rank;
    }
    for (const auto& [m, t] : bias_diffs) {
        const Tensor* b = lora_bias(m);
        auto* dst = reinterpret_cast<float*>(static_cast<char*>(b->buf->contents()) + b->off);
        const auto d = read_floats(*file, *t);
        for (size_t i = 0; i < d.size(); ++i)
            dst[i] += scale * d[i];
        ++done;
    }
    for (const auto& [m, t] : weight_diffs) {
        uint32_t r = 0, k = 0;
        float* w = lora_cpu(m, r, k);
        const auto d = read_floats(*file, *t);
        for (size_t i = 0; i < d.size(); ++i)
            w[i] += scale * d[i];
        ++done;
    }
    if (!done)
        throw std::runtime_error(path + ": no LoRA weights in it");
    lora_merged();
    log::info("LoRA {}: {} tensors adapted ({} kept unmerged; rank {}, alpha {}), scale {}", path, done, side,
              rank_seen, alpha ? std::format("{}", *alpha) : std::string("= rank"), scale);
    return done;
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
