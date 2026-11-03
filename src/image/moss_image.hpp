#pragma once

#include "image/engine.hpp"

namespace mako::image {

// MOSS-Image: the last hidden state of a Qwen3.5 language model, refined by two blocks, conditions a single-stream
// transformer over [image, text] whose blocks share one timestep modulation; guided by default against an empty
// negative prompt whose pass skips blocks (one by default).
class MossImage final : public Engine {
public:
    MossImage(const std::string& dir, mtl::Context& ctx, const Options& opts = {});

    const char* name() const override { return "MOSS-Image"; }
    uint32_t default_steps() const override { return ds_.codes.empty() ? 50 : uint32_t(ds_.codes.size()); }
    float default_guidance() const override { return ds_.codes.empty() ? 3.0f : 1.0f; }
    std::optional<std::string> default_negative() const override { return ""; }
    std::vector<uint32_t> default_skip_blocks() const override { return {22}; }
    // The text encoder's final hidden states of the templated prompt, [S][4096].
    std::vector<float> prompt_embeddings(const std::string& prompt) override;
    // Refines the text rows once; every step then runs the whole [image, text] sequence. Slot 1 (the negative
    // prompt's) skips the blocks set (by default the one the reference skips there).
    void begin(std::span<const float> prompt, uint32_t h, uint32_t w, uint32_t slot = 0) override;
    std::vector<float> step(std::span<const float> latents, float sigma, uint32_t slot = 0) override;
    // The shifted linear schedule snapped to the 1000 training noise levels (repeats dropped).
    std::vector<float> sigmas(uint32_t steps, uint32_t tokens, std::optional<float> terminal) const override;
    float top_sigma(uint32_t tokens) const override;

    // The chat template around a prompt (a user turn, the assistant's opened with thinking).
    static std::string templated(const std::string& prompt);

protected:
    LoraTarget lora_target(const std::string& module) override;
    void set_skip_blocks(std::span<const uint32_t> blocks) override;
    // The blocks' projections keep LoRAs unmerged: distillation LoRAs change them by less than BF16's rounding.
    bool lora_side(const std::string& module, std::span<const float> a, std::span<const float> b, uint32_t rank,
                   float s) override;
    void lora_merged() override;
    bool keeps_loras_unmerged() const override { return true; }
    const DataSpace* data_space() const override { return ds_.codes.empty() ? nullptr : &ds_; }
    bool has_control() const override { return ctrl_; }
    void set_control(std::vector<float> color, std::vector<float> depth) override;
    void control_on(bool color, bool depth) override;

private:
    // q | k | v in one matrix; the FFN's activation input and its gate in another.
    struct Block {
        Tensor q, o, norm_q, norm_k, up_gate, down;
    };
    // The text refiner's blocks: biases throughout, ungated residuals.
    struct TextBlock {
        Tensor norm1, norm2, qkv, qkv_b, norm_q, norm_k, o, o_b, gate_up, gate_up_b, down, down_b;
    };
    void load_transformer(const safetensors::Files& f);
    void load_adapter(const std::string& dir);
    void load_control(const std::string& path);
    // Rotary angles for an h x w image followed by S text tokens: (0, S, y, x) and (0, t, 0, 0), image centred.
    void rope_table(MTL::Buffer* cs, uint32_t S, uint32_t h, uint32_t w) const;

    // With a control adapter, img_in_ takes [latents | colour map | depth map] (3 x 128 per token).
    bool ctrl_ = false, color_on_ = false, depth_on_ = false;
    std::vector<float> ctrl_color_, ctrl_depth_; // the maps' latents, [N][128] each; empty: none
    Tensor img_in_, img_in_b_, txt_proj_, txt_proj_b_, t1_, t1_b_, t2_, t2_b_, mod_, mod_b_, head_mod_, head_mod_b_;
    Tensor out_norm_, out_, norms_; // norms_: each block's two norm scales, [blocks][2][D]
    std::vector<Block> blocks_;
    // A projection's unmerged LoRAs: each file's factors side by side, down [R][k] and up [n][R] (BF16, R padded to
    // 64), from the pieces read until lora_merged().
    struct Side {
        struct Piece {
            uint32_t row0, rows, rank;
            std::vector<float> a, b; // b scaled
        };
        struct Group {
            Tensor down, up;
            uint32_t R = 0;
        };
        std::vector<Piece> pieces;
        std::vector<Group> groups;
    };
    enum { kSideQkv, kSideO, kSideUpGate, kSideDown, kSides };
    std::vector<std::array<Side, kSides>> side_; // per block
    // An adapter's F32 time path, run on the CPU: time_embedding.0 [3072][256] and .2 [3072][3072] with biases.
    std::vector<float> t1f_, t1bf_, t2f_, t2bf_;
    DataSpace ds_;
    std::vector<TextBlock> text_blocks_;
    float bands_[16] = {};
    static constexpr uint32_t kDim = 3072, kHeads = 24, kText = 4096, kTimeDim = 256, kLatentWidth = 128;
    static constexpr float kEps = 1e-6f, kShift = 5.0f;
    // The generation begun: the latent grid and each slot's text rows (refined into k_txt<slot>).
    uint32_t gh_ = 0, gw_ = 0, S_[2] = {};
    std::vector<uint32_t> skip_ = default_skip_blocks(); // the blocks slot 1 leaves out
};

} // namespace mako::image
