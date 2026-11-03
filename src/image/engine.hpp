#pragma once

#include "gpu/runtime.hpp"
#include "image/png.hpp"
#include "safetensors/safetensors.hpp"
#include "text/tokenizer.hpp"

#include <functional>
#include <memory>
#include <optional>

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace mako::image {

struct Options {
    std::string text_encoder; // another directory of the text encoder's safetensors (the same model's, changed)
    std::string transformer;  // another file or directory of the transformer's weights
    // A data-space distillation adapter's directory (MOSS-Image): lora.safetensors (kept unmerged),
    // time_embedding.safetensors (F32 time path) and dataspace.safetensors (its sampler).
    std::string adapter;
    gpu::Telemetry telemetry = gpu::Telemetry::off;
};

// A LoRA file, its scale, and whether it is merged into the weights or kept beside them (Engine::add_lora).
struct Lora {
    std::string path;
    float scale = 1.0f;
    std::optional<bool> merge; // unset: the model's way
    bool operator==(const Lora&) const = default;
};
// LoRAs, added in order.
using Loras = std::vector<Lora>;

// How a step moves the latents: Euler, Euler ancestral (towards a lower noise level, then fresh noise up to the
// next; models that predict the noise only), or ER-SDE (image::ErSde).
enum class Sampler { euler, euler_ancestral, er_sde };
// "euler", "euler_a" (also "euler_ancestral") or "er_sde"; nullopt for another name.
std::optional<Sampler> parse_sampler(std::string_view name);
// The name parse_sampler() reads.
const char* sampler_name(Sampler s);

// An image model from its original files (a directory with model_index.json, text_encoder/, transformer/, vae/,
// scheduler/): the parts the models share, the Qwen3-VL text encoder, the VAE decoder, the flow-matching schedule and
// the denoising loop. QwenImage and Krea2 add their diffusion transformers; open() picks one from model_index.json.
class Engine : public gpu::Runtime {
public:
    // The model in `dir`, by the pipeline model_index.json names. Throws on a model it does not know.
    static std::unique_ptr<Engine> open(const std::string& dir, mtl::Context& ctx, const Options& opts = {});
    // Which model open() would load from `dir` ("" if none it knows), without loading it.
    static std::string family(const std::string& dir);
    // Whether a family's models predict the noise (predicts_noise()).
    static bool predicts_noise(const std::string& family) { return family == "SDXL"; }
    // Whether a family's negative prompt pass can leave blocks out (Request::skip_blocks).
    static bool skips_blocks(const std::string& family) { return family == "MOSS-Image"; }
    static bool keeps_loras_unmerged(const std::string& family) { return family == "MOSS-Image"; }
    // Whether the family's VAE encodes images (image to image): the ones encode() takes.
    static bool encodes_images(const std::string& family) { return family == "MOSS-Image"; }

    virtual ~Engine() = default;

    // A model's parts where they are not laid out as Diffusers has them under one directory: a file or directory
    // each, the scheduler's config and the VAE's (config.json beside it) optional, the text encoder's rotary base
    // where it has no config (0: its default).
    struct Parts {
        std::string text_encoder, vae, scheduler, tokenizer;
        float text_theta = 0;
    };
    virtual const char* name() const = 0;
    virtual uint32_t default_steps() const = 0;
    // The guidance scale with a negative prompt: the step's velocity is neg + scale * (pos - neg).
    virtual float default_guidance() const = 0;
    // The negative prompt of requests that give none (unset: no guidance then).
    virtual std::optional<std::string> default_negative() const { return std::nullopt; }
    virtual Sampler default_sampler() const { return Sampler::euler; }
    // step() predicts the noise (the latents are image + sigma * noise, the start noise scaled to the first level)
    // rather than a flow's velocity (latents (1 - sigma) * image + sigma * noise).
    virtual bool predicts_noise() const { return false; }
    // The blocks the negative prompt's pass leaves out by default (only models that skip blocks have any).
    virtual std::vector<uint32_t> default_skip_blocks() const { return {}; }

    struct Request {
        std::string prompt;
        uint32_t width = 1024, height = 1024; // multiples of 32
        uint32_t steps = 0;                   // 0: the model's default
        uint64_t seed = 0;
        std::span<const float> noise; // instead of the seed's: [h * w][latent_width()]
        std::optional<float> terminal; // the schedule's last noise level (0: not stretched); unset: the model's
        // What the image should not be: classifier-free guidance, a second transformer pass a step (unset: none, or
        // the model's own); none at scale 1.
        std::optional<std::string> negative;
        std::optional<float> guidance; // its scale (unset: the model's)
        std::optional<Sampler> sampler; // unset: the model's
        // Transformer blocks the negative prompt's pass leaves out (zero-based; skips_blocks() models only); unset:
        // the model's.
        std::optional<std::vector<uint32_t>> skip_blocks;
        // Image to image: the image to start from, RGBA [height][width][4] in [-1, 1], noised to the level the
        // schedule is `strength` of the way up from its end, and only the steps from there run (1: from the top).
        std::span<const float> image;
        float strength = 1.0f;
        std::function<void(uint32_t step, uint32_t steps)> progress;
    };
    // Image to image: the first of `steps` steps to run at `strength` (the share run, rounded to whole steps; below 1
    // at least one left out, at most all but one).
    static uint32_t start_step(uint32_t steps, float strength);
    // Text (and image) to image: RGBA [height][width][4] in [-1, 1].
    std::vector<float> generate(const Request& r);
    // What generate() makes of r (its steps, guidance and negative prompt in effect), for the image's metadata; the
    // caller adds the model's name and LoRAs.
    Provenance provenance(const Request& r) const;

    // The transformer's text input for a prompt.
    virtual std::vector<float> prompt_embeddings(const std::string& prompt) = 0;
    // A generation's text and latent grid (h x w tokens of latent_width()), then step() for each denoising step: the
    // velocity the transformer predicts for latents [h * w][latent_width()] at noise level sigma (or the noise, with
    // predicts_noise()). Slots keep separate texts (1: the negative prompt's) over the same grid.
    virtual void begin(std::span<const float> prompt, uint32_t h, uint32_t w, uint32_t slot = 0) = 0;
    virtual std::vector<float> step(std::span<const float> latents, float sigma, uint32_t slot = 0) = 0;

    // The text encoder over token ids: rows [drop, n) of its last layer's output before the final norm ([n - drop][E])
    // or after it (normed), or with taps, of the hidden states listed (0: the embeddings, i: layer i's output),
    // [n - drop][taps][E].
    std::vector<float> encode_text(std::span<const int32_t> ids, uint32_t drop, std::span<const uint32_t> taps = {},
                                   bool normed = false);
    uint32_t text_width() const { return tE_; }

    // The VAE decoder: transformer latents [h * w][latent_width()] (normalized) to RGBA [16h][16w][4] in [-1, 1].
    std::vector<float> decode(std::span<const float> latents, uint32_t h, uint32_t w);
    // The VAE encoder: RGBA [16h][16w][4] in [-1, 1] (alpha unused) to transformer latents [h * w][latent_width()]
    // (normalized), the posterior's mean. Its weights load with the first call. AutoencoderKL VAEs in Diffusers' names
    // only (MOSS-Image's); throws for the others.
    std::vector<float> encode(std::span<const float> rgba, uint32_t h, uint32_t w);
    // A transformer token's latent values: the VAE's channels times the pixels of its patch.
    uint32_t latent_width() const { return lat_; }

    // The noise levels of `steps` denoising steps for an image of `tokens` latent tokens, then 0 (the schedule's
    // shift, stretched to end at its terminal value). A model's own schedule may have fewer.
    virtual std::vector<float> sigmas(uint32_t steps, uint32_t tokens,
                                      std::optional<float> terminal = std::nullopt) const;
    // A flow model's noise level a hundredth of a percent of the way down its schedule's shift, where ER-SDE starts in
    // place of 1.
    virtual float top_sigma(uint32_t tokens) const;

    // Adds a LoRA to the transformer for good: W += scale * (alpha / rank) * B A for each matrix it adapts, merged
    // (rounded back to the weights' BF16) or, where the model can (keeps_loras_unmerged()), kept beside the weight and
    // applied on every pass; b += scale * diff_b for whole bias differences. merge unset: the model's way (unmerged
    // where it can). Keys as Diffusers, PEFT or ComfyUI name them, or Kohya's (lora_unet_ and the module path with
    // underscores); alpha from the adapter's .alpha tensors, else its PEFT config in the metadata, else the rank.
    // Returns the tensors adapted; throws on a file it cannot use (keys it does not know are an error).
    uint32_t add_lora(const std::string& path, float scale = 1.0f, std::optional<bool> merge = std::nullopt);
    uint32_t add_lora(const Lora& l) { return add_lora(l.path, l.scale, l.merge); }
    // Whether LoRAs can stay unmerged (Lora::merge false): the family's (keeps_loras_unmerged(family)).
    virtual bool keeps_loras_unmerged() const { return false; }
    const text::Tokenizer& tokenizer() const { return *tok_; }

protected:
    // Which blocks the negative prompt's pass leaves out from here on; throws on blocks the model does not have, or
    // any for a model that skips none.
    virtual void set_skip_blocks(std::span<const uint32_t> blocks);
    // dir's text encoder (or opts.text_encoder), VAE, scheduler and tokenizer (processor/ or tokenizer/).
    Engine(const std::string& dir, mtl::Context& ctx, const Options& opts);
    Engine(const Parts& parts, mtl::Context& ctx, const Options& opts);
    // Neither text encoder nor VAE: the model loads its own (load_vae_ldm for the VAE).
    Engine(mtl::Context& ctx, const Options& opts);
    static Parts diffusers_parts(const std::string& dir, const Options& opts);
    // The VAE (4 latent channels, 8x) under `prefix` in a single-file checkpoint, in the original layout's names;
    // a token packs a 2x2 patch, the latents scaled by `scale`.
    void load_vae_ldm(const std::string& file, const std::string& prefix, float scale);

    // Convolutions as GEMMs over channels-last images: weights F16 [cout][kpad] (cin * k * k columns, a 3x3's zero
    // padded to a multiple of 64). A causal 3D convolution's weights are its last time tap's (a single frame).
    struct Conv {
        Tensor w, b;
        uint32_t cin = 0, cout = 0, k = 3, kpad = 0, stride = 1;
        uint32_t pad = 1; // stride 2: padding above and left (below and right always 1)
    };
    // `name`.weight and .bias (F32, F16 or BF16): a convolution (or a linear layer, as a 1x1), or several of one
    // input's with their outputs side by side.
    Conv load_conv(const safetensors::Files& files, std::span<const std::string> names, uint32_t k);
    // y = conv(x) over an h x w image (with stride 2, a 3x3 to h / 2 x w / 2).
    void conv(MTL::ComputeCommandEncoder*& enc, const Conv& c, MTL::Buffer* x, uint32_t h, uint32_t w, MTL::Buffer* y);
    // Group norm (32 groups) times g plus b over channels-last images, then SiLU when silu is set.
    void group_norm(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* x, const Tensor& g, const Tensor& b, MTL::Buffer* y,
                    uint32_t rows, uint32_t c, bool silu, float eps = 1e-6f);

    // The weight a LoRA module adapts: its matrix and the rows it has there (fused matrices share one).
    struct LoraTarget {
        Tensor* w = nullptr;
        uint32_t row0 = 0, rows = 0;
    };
    // `module` with the file's prefixes stripped (transformer., diffusion_model., ...).
    virtual LoraTarget lora_target(const std::string& module) = 0;
    // The F32 bias of `module`, for an adapter's whole bias difference (null: none).
    virtual Tensor* lora_bias(const std::string&) { return nullptr; }
    // A small matrix of `module` the model keeps on the CPU, [rows][cols] floats (null: none).
    virtual float* lora_cpu(const std::string&, uint32_t& /*rows*/, uint32_t& /*cols*/) { return nullptr; }
    // After a LoRA is merged (what the model computed from the weights before is stale).
    virtual void lora_merged() {}
    // Keeps W += s * B A for `module` unmerged, applied beside W on every pass (a: [rank][k], b: [rows][rank]); false:
    // merge it. For changes smaller than the weights' rounding, which merging into them would lose.
    virtual bool lora_side(const std::string& /*module*/, std::span<const float> /*a*/, std::span<const float> /*b*/,
                           uint32_t /*rank*/, float /*s*/)
    {
        return false;
    }
    // A data-space sampler (an adapter's): each step runs the model on z + Q x0 at the next code's timestep and takes
    // x0 = z - its output, from one fixed noise z and x0 = 0; no guidance.
    struct DataSpace {
        std::vector<float> q;     // [latent_width()][latent_width()], output channel first
        std::vector<float> codes; // the steps' timesteps (1000: all noise)
    };
    virtual const DataSpace* data_space() const { return nullptr; }

    Tensor upload(const safetensors::Files& files, const std::string& name, bool matrix);
    // A tensor's values as F32 (from F32, F16 or BF16).
    Tensor upload_f32(const safetensors::Files& files, const std::string& name);
    // BF16 (or F32, rounded to BF16) matrices of one width, their rows one after another in one tensor; F16 ones stay
    // F16.
    Tensor upload_rows(const safetensors::Files& files, std::span<const std::string> names);
    // An RMS norm's scale stored zero-centred: weight + 1, as floats.
    Tensor upload_plus_one(const safetensors::Files& files, const std::string& name);
    void add(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* x, MTL::Buffer* z, MTL::Buffer* y, uint32_t n);
    // Per-head q / k / v half tiles of T rows (rope: 0 none, 1 halves, 2 adjacent pairs; cs: the angle table).
    void heads(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* x, uint32_t ld, uint32_t col0, uint32_t heads,
               uint32_t T, uint32_t n_pad, MTL::Buffer* cs, uint32_t rope, MTL::Buffer* y);
    // heads() with each head RMS-normed first (times w).
    void heads_norm(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* x, uint32_t ld, uint32_t col0, uint32_t heads,
                    uint32_t T, uint32_t n_pad, MTL::Buffer* cs, uint64_t cs_off, uint32_t rope, const Tensor& w,
                    float eps, uint32_t row0, MTL::Buffer* y);
    // block: positions from here on see each other both ways; below it, causal (0: all both ways).
    void attention(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* q, MTL::Buffer* k, MTL::Buffer* v, MTL::Buffer* o,
                   uint32_t n_q, uint32_t q_pad, uint32_t n_k, uint32_t k_pad, uint32_t heads, uint32_t kv_heads,
                   uint32_t q_pos0, uint32_t block);
    // y[r][i] += b[i] over rows of n.
    void bias(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* y, const Tensor& b, uint32_t n, uint32_t rows,
              uint64_t y_off = 0);
    // A tensor's values as floats.
    static std::vector<float> floats(const safetensors::Files& files, const std::string& name);
    void unary(MTL::ComputeCommandEncoder*& enc, const char* fn, MTL::Buffer* x, uint32_t n);

    static constexpr uint32_t kHeadDim = 128;
    static constexpr uint32_t kAttnRows = 64; // queries a threadgroup of the attention kernel
    static constexpr uint32_t kLatent = 64;   // a transformer token's latent values (the Qwen-Image VAEs')

    std::optional<text::Tokenizer> tok_;
    uint32_t lat_ = kLatent;
    struct Schedule {
        double base_seq = 256, max_seq = 4096, base_shift = 0.5, max_shift = 1.15, terminal = 0;
        std::optional<double> mu; // a fixed shift (distilled models) instead of one by image size
    } sched_;
    double shift_mu(uint32_t tokens) const; // the schedule's shift for an image of `tokens` tokens
    std::string dir_;

private:
    struct TextLayer {
        Tensor attn_norm, q, k, v, o, q_norm, k_norm, ffn_norm, gate, up, down;
        // A gated delta rule layer (Qwen3.5's linear attention): q | k | v in one matrix (q), its causal convolution,
        // the output gate z, b | a for the update and decay, the decay's parameters; o its output, q_norm the gated
        // norm's scale.
        bool linear = false;
        Tensor conv, z, ba, a_log, dt_bias;
    };
    void load_text(const std::string& tdir);
    // A gated delta rule layer and a gated attention layer (Qwen3.5) from xn, into y.
    void text_linear(MTL::ComputeCommandEncoder*& enc, const TextLayer& L, MTL::Buffer* xn, MTL::Buffer* y, uint32_t T);
    void text_gated_attention(MTL::ComputeCommandEncoder*& enc, const TextLayer& L, MTL::Buffer* xn, MTL::Buffer* y,
                              uint32_t T, MTL::Buffer* cs);
    // Rows of the token embedding table (BF16, read from the file).
    void embed(std::span<const int32_t> ids, float* out) const;

    struct Res {
        Tensor g1, g2; // the norms' scales (and with group norms, b1, b2 their biases)
        Tensor b1, b2;
        Conv c1, c2, sc;
        bool shortcut = false;
    };
    struct Up {
        std::vector<Res> res;
        bool up = false;
        Conv conv;
        uint32_t cin = 0, cout = 0, ft = 1;
    };
    void load_vae();
    // The AutoencoderKL layout (FLUX.2's and SDXL's): group norms, separate q, k, v, upsamplers' plain convolutions.
    void load_vae_kl();
    Res load_res_kl(const std::string& prefix);
    // The mid block's single-head attention over an h x w image, added to x; t1, t2, t3 scratch.
    void vae_attention(MTL::ComputeCommandEncoder*& enc, const Tensor& g, const Tensor* b, const Conv& qkv,
                       const Conv& proj, MTL::Buffer* x, uint32_t h, uint32_t w, MTL::Buffer* t1, MTL::Buffer* t2,
                       MTL::Buffer* t3);
    struct Down {
        std::vector<Res> res;
        bool down = false;
        Conv conv;
    };
    struct VaeEncoder {
        Conv in, out, quant, qkv, proj;
        Res mid0, mid1;
        Tensor attn_g, attn_b, out_g, out_b;
        std::vector<Down> downs;
    };
    void load_encoder_kl();
    std::unique_ptr<VaeEncoder> venc_;
    // A VAE tensor's name in the file (the Diffusers layout's, or the original one's).
    std::string vk(const std::string& name) const;
    // A VAE convolution by its Diffusers name.
    Conv load_conv(const std::string& name, uint32_t k);
    Conv load_conv(std::span<const std::string> names, uint32_t k);
    // y = conv2(silu(norm2(conv1(silu(norm1(x)))))) + shortcut(x); t1, t2 scratch.
    void res_block(MTL::ComputeCommandEncoder*& enc, const Res& r, MTL::Buffer* x, uint32_t h, uint32_t w,
                   MTL::Buffer* y, MTL::Buffer* t1, MTL::Buffer* t2);
    void norm_silu(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* x, const Tensor& g, MTL::Buffer* y, uint32_t rows,
                   uint32_t c, const Tensor* b = nullptr);

    safetensors::Files text_files_, vae_files_;
    safetensors::Files::Ref embd_;
    std::vector<TextLayer> text_;
    Tensor text_norm_;
    uint32_t tE_ = 0, tHeads_ = 0, tKv_ = 0;
    float tEps_ = 1e-6f, tTheta_ = 5e6f;
    // Qwen3.5: zero-centred norms, attention heads tHead_ wide with an output gate and the first tRot_ values rotated,
    // and the delta rule's key and value heads (of 128).
    bool qwen35_ = false;
    uint32_t tHead_ = kHeadDim, tRot_ = kHeadDim, tKeyHeads_ = 0, tValueHeads_ = 0;

    Conv post_quant_, conv_in_, conv_out_, qkv_, attn_proj_;
    Tensor attn_g_, out_g_, attn_b_, out_b_;
    Res mid0_, mid1_;
    std::vector<Up> ups_;
    // Each token latent value's mean and deviation (the VAE's per channel, or its batch norm's per token value).
    std::vector<float> lat_mean_, lat_std_;
    // The VAE's latent channels and the 2x2 patches a token packs (16 channels: Qwen-Image's 8x VAE), and whether its
    // up blocks have the duplicating shortcut (Qwen-Image 2.1's).
    uint32_t z_ = kLatent, patch_ = 1;
    bool residual_ = false;
    bool kl_ = false;  // the AutoencoderKL layout (load_vae_kl)
    bool wan_ = false; // the first Qwen-Image VAE in its original names
    std::string ldm_;  // the prefix of a VAE in the original AutoencoderKL names (SDXL checkpoints)
    unsigned ldm_levels_ = 0;
};

// A prompt's parts and their weights: "(text)" times 1.1, "(text:w)" w, nested ones multiplied, "\(" and "\)"
// literal parentheses.
std::vector<std::pair<std::string, float>> prompt_weights(const std::string& prompt);

// "FILE", "FILE:SCALE", then optionally ":merge" or ":unmerged" (the scale a number after its colon). Throws
// std::invalid_argument.
Lora parse_lora(const std::string& spec);

} // namespace mako::image
