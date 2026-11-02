#pragma once

#include "image/clip_tokenizer.hpp"
#include "image/engine.hpp"

#include <array>
#include <unordered_map>

namespace mako::image {

// SDXL from a single-file checkpoint in the original names (model.diffusion_model., conditioner.embedders.0 / 1.,
// first_stage_model.) with SDXL's tokenizer/ beside it. Two text encoders, CLIP ViT-L and OpenCLIP bigG, give their
// second to last layers' states side by side, and bigG's projected final state at the end token with the image size
// is the UNet's added conditioning. The UNet predicts the noise of a discrete schedule (1000 levels, scaled linear
// betas); its timestep is the nearest level's. The VAE is AutoencoderKL's with 4 channels.
class Sdxl final : public Engine {
public:
    Sdxl(const std::string& dir, mtl::Context& ctx, const Options& opts = {});
    // The checkpoint in dir (its one .safetensors with SDXL's UNet and both text encoders), or "".
    static std::string checkpoint(const std::string& dir);

    const char* name() const override { return "SDXL"; }
    uint32_t default_steps() const override { return 28; }
    float default_guidance() const override { return 6.0f; }
    std::optional<std::string> default_negative() const override;
    Sampler default_sampler() const override { return Sampler::euler_ancestral; }
    bool predicts_noise() const override { return true; }

    // Packed: S, the context [S][2048], the pooled state [1280].
    std::vector<float> prompt_embeddings(const std::string& prompt) override;
    void begin(std::span<const float> prompt, uint32_t h, uint32_t w, uint32_t slot = 0) override;
    std::vector<float> step(std::span<const float> latents, float sigma, uint32_t slot = 0) override;
    // Levels evenly spaced in timestep from the top to the bottom (interpolated in log sigma), then 0.
    std::vector<float> sigmas(uint32_t steps, uint32_t tokens, std::optional<float> terminal) const override;

    struct Token {
        int32_t id;
        float weight;
    };
    // The prompt in chunks of 77: the start token, up to 75 of the prompt's, the end token, then `pad`. Each
    // weighted part of the prompt stays in one chunk when it is shorter than 8 tokens; longer ones are cut where a
    // chunk is full.
    std::vector<std::vector<Token>> chunks(const std::string& prompt, int32_t pad) const;
    // One chunk through text encoder `which` (0: ViT-L, 1: bigG): the second to last layer's states [77][E]; with
    // `pooled`, bigG's final normed state at the first end token, projected.
    std::vector<float> encode_clip(int which, std::span<const int32_t> ids, std::vector<float>* pooled = nullptr);
    // The level nearest sigma (in log sigma).
    uint32_t timestep(float sigma) const;

    static constexpr int32_t kPadL = ClipTokenizer::kEnd, kPadG = 0;

protected:
    // UNet modules by their original names (input_blocks.4.1.transformer_blocks.0.attn1.to_q, ...), the text
    // encoders' as Kohya names them (lora_te1_ / lora_te2_ and Hugging Face's CLIP names); "_" and "." alike.
    LoraTarget lora_target(const std::string& module) override;
    void lora_merged() override;

private:
    struct ClipLayer {
        Tensor ln1g, ln1b, qkv, qkv_b, o, o_b, ln2g, ln2b, fc1, fc1_b, fc2, fc2_b;
    };
    struct Clip {
        std::vector<ClipLayer> layers;
        uint32_t E = 0, layers_total = 0;
        bool quick_gelu = false;
        safetensors::Files::Ref tok;
        std::vector<float> pos, lnf_g, lnf_b, proj; // proj [E][out] as stored (bigG)
    };
    struct Res {
        Tensor n1g, n1b, n2g, n2b;
        Conv c1, c2, skip;
        bool has_skip = false;
        uint32_t emb_off = 0; // its rows of the timestep embedding's projection
    };
    struct TBlock {
        Tensor n1g, n1b, n2g, n2b, n3g, n3b, qkv, o1, o1b, q2, kv2, o2, o2b, ff1, ff1b, ff2, ff2b;
    };
    struct Trans {
        Tensor ng, nb, pin, pinb, pout, poutb;
        std::vector<TBlock> blocks;
        uint32_t c = 0, first = 0; // channels; its first block's index among all
    };
    // An input block (conv_in, a stride-2 convolution, or a residual block with a transformer), the middle's parts,
    // an output block (a residual block, a transformer, an upsampling convolution).
    struct Block {
        enum Kind { conv_in, down, res } kind = res;
        Conv conv;
        Res r;
        Trans t;
        bool has_t = false, up = false;
    };

    // An F16 matrix as it is (others as upload() makes them).
    Tensor matrix(const std::string& name);
    void load_clip(int which);
    void load_unet();
    Res load_res(const std::string& p, std::vector<std::string>& emb_w, std::vector<std::string>& emb_b);
    Trans load_trans(const std::string& p);

    void ln(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* x, const Tensor& g, const Tensor& b, MTL::Buffer* y,
            uint32_t rows, uint32_t n);
    // 64-wide half head tiles of T rows of x (columns col0 + h * 64 of rows ld floats apart), padded rows zeroed.
    void heads64(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* x, uint32_t ld, uint32_t col0, uint32_t n_heads,
                 uint32_t T, uint32_t n_pad, MTL::Buffer* y);
    void attend64(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* q, MTL::Buffer* k, MTL::Buffer* v, MTL::Buffer* o,
                  uint32_t n_q, uint32_t n_k, uint32_t n_heads, uint32_t block);
    void res_fwd(MTL::ComputeCommandEncoder*& enc, const Res& r, MTL::Buffer* x, uint32_t h, uint32_t w,
                 MTL::Buffer* y, MTL::Buffer* emb);
    void trans_fwd(MTL::ComputeCommandEncoder*& enc, const Trans& t, MTL::Buffer* x, uint32_t n, MTL::Buffer* y,
                   uint32_t slot);

    safetensors::Files files_;
    std::optional<ClipTokenizer> clip_tok_;
    std::array<Clip, 2> clip_;
    // Unweighted states of the empty prompt (start, end, padding) per encoder, for weighted tokens.
    std::array<std::vector<float>, 2> empty_;

    std::vector<Block> in_, out_;
    Res mid0_, mid1_;
    Trans mid_t_;
    Tensor t1_, t1b_, t2_, t2b_, l1_, l1b_, l2_, l2b_, emb_w_, emb_b_, out_g_, out_b_;
    Conv out_c_;
    uint32_t emb_rows_ = 0, n_tblocks_ = 0;

    std::array<float, 1000> sigma_{}, log_sigma_{};
    // Per slot: the added conditioning (pooled state and sizes, 2816) and the context's length.
    std::array<std::vector<float>, 2> y_;
    std::array<uint32_t, 2> ctx_rows_{};
    // LoRA targets by module name with "." for "_".
    std::unordered_map<std::string, LoraTarget> lora_;
    uint32_t gh_ = 0, gw_ = 0;
};

} // namespace mako::image
