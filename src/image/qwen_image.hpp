#pragma once

#include "image/engine.hpp"
#include "models/qwen_vision.hpp"

namespace mako::image {

// Qwen-Image 2.1: the text encoder's last layer (a Qwen3-VL language model) conditions a single-stream transformer.
// Reference images (editing) go in twice: through the vision tower into the prompt, and as latents in the sequence
// where their vision tokens were, modulated at t = 0 as the text is; a text row sees the rows before it, a
// reference's rows everything before their block's end, the generated image's rows everything. The first step runs
// that prefix and keeps its keys and values, later steps the image's rows only.
class QwenImage final : public Engine {
public:
    QwenImage(const std::string& dir, mtl::Context& ctx, const Options& opts = {});

    const char* name() const override { return "Qwen-Image 2.1"; }
    uint32_t default_steps() const override { return 40; }
    // The reference pipeline runs without guidance by default; 4 as Qwen-Image's first release suggested (2 to 6
    // tried: clean, the negative prompt followed).
    float default_guidance() const override { return 4.0f; }
    std::vector<float> prompt_embeddings(const std::string& prompt) override;
    // The prompt embeddings [S][4096]. The first step runs the whole sequence and keeps the prefix's keys and
    // values, the later ones only the image's.
    void begin(std::span<const float> prompt, uint32_t h, uint32_t w, uint32_t slot = 0) override;
    std::vector<float> step(std::span<const float> latents, float sigma, uint32_t slot = 0) override;

    // The prompt template around `prompt`, and the token count of its system part.
    static std::string templated(const std::string& prompt);
    uint32_t system_tokens() const { return drop_; }
    // The size a reference is taken at: about 1024 x 1024 pixels' area at its aspect ratio, sides multiples of 32.
    static std::pair<uint32_t, uint32_t> reference_size(uint32_t w, uint32_t h);
    // The references as set_references() encoded them: their count, and each one's latents ([h * w][64],
    // normalized) and latent grid.
    size_t references() const { return refs_.size(); }
    std::span<const float> reference_latents(size_t i) const { return refs_[i].latents; }
    std::pair<uint32_t, uint32_t> reference_grid(size_t i) const { return {refs_[i].lh, refs_[i].lw}; }
    const models::QwenVision::Features& reference_features(size_t i) const { return refs_[i].features; }
    // The token ids the last prompt_embeddings() encoded (the template with the pictures' vision blocks).
    const std::vector<int32_t>& presentation() const { return ids_; }
    void set_references(const Request& r) override;
    // For the tests: a reference's latents from the reference pipeline in place of the engine's encoder's.
    void set_reference_latents(size_t i, std::vector<float> latents);

protected:
    LoraTarget lora_target(const std::string& module) override;

private:
    struct Block {
        Tensor qkv, o, norm_q, norm_k, gate_up, down; // q, k, v and gate, up side by side: one GEMM each
    };
    void load_transformer(const safetensors::Files& f);
    models::QwenVision& vision();

    Tensor img_in_, txt_norm_, txt_in1_, txt_in2_, t_lin1_, t_lin2_, mod_, out_lin_, proj_out_;
    std::vector<Block> blocks_;
    static constexpr uint32_t kDim = 4096, kHeads = 32, kTimeDim = 256;
    static constexpr uint32_t kMaxPrompt = 1024; // tokens after the system part (text to image)
    static constexpr uint32_t kReferenceArea = 1024 * 1024;
    static constexpr float kEps = 1e-6f;
    static constexpr const char* kSystem = "Comprehend and analyze the provided prompt.";
    uint32_t drop_ = 0;
    std::string vision_dir_;
    std::unique_ptr<models::QwenVision> vision_;
    struct Ref {
        uint32_t lh = 0, lw = 0;    // its latent grid
        std::vector<float> latents; // [lh * lw][64], normalized
        models::QwenVision::Features features;
    };
    std::vector<Ref> refs_;
    std::vector<uint32_t> slots_; // per reference, the prompt row its latents go before (prompt_embeddings() sets them)
    std::vector<int32_t> ids_;
    // The generation begun: the latent grid; for each slot, its prompt rows, the prefix (the prompt's rows with each
    // reference's latent rows at its slot) as segments of the sequence, steps run and the prefix's keys and values of
    // each block (half [heads][P][128]).
    uint32_t gh_ = 0, gw_ = 0;
    struct Segment {
        bool image;            // a reference's latent rows, else prompt rows
        uint32_t row0, n, src; // its first sequence row and rows; its first prompt row, or the reference
    };
    struct Text {
        uint32_t S = 0, P = 0, steps = 0;
        std::vector<float> prompt;
        std::vector<Segment> segments;
        std::vector<mtl::Ref<MTL::Buffer>> k, v;
    } texts_[2];
};

} // namespace mako::image
