#pragma once

#include "image/engine.hpp"
#include "models/h3_vae.hpp"
#include "models/h3.hpp"
#include "models/qwen_vision.hpp"

namespace mako::image {

// MiniMax-H3-Image: MiniMax-H3's transformer without its audio stream over one packed sequence [text | reference
// images | image], conditioned on the prompt verbatim through the Qwen3-VL-32B encoder's 50th layer; six Euler
// steps on a fixed grid, each with its own output head (the checkpoint is distilled: no guidance, no other step
// count); its own VAE. An image is generated on a canvas of at least 2048x2048 pixels' area at the aspect ratio
// asked for and resized to the size asked for; reference images (edits) go into the sequence at their own size,
// held near clean, and before the prompt as a "Picture i: " label and a vision block each: the prompt names them
// <Picture 1>, <Picture 2>, ...
class H3Image final : public Engine {
public:
    H3Image(const std::string& dir, mtl::Context& ctx, const Options& opts = {});

    const char* name() const override { return "MiniMax-H3-Image"; }
    uint32_t default_steps() const override { return uint32_t(grid_.size() - 1); }
    float default_guidance() const override { return 1.0f; }
    // The text encoder's 50th layer over the presentation: the references' labels and vision blocks, then the
    // prompt's tokens (no template, at most 4096), [S][5120].
    std::vector<float> prompt_embeddings(const std::string& prompt) override;
    // Refines the text rows once (the transformer's token refiner); the references' rows go ahead of the h x w
    // generated latents' 2x2 patches. Slot 0 only: the model is not guided.
    void begin(std::span<const float> prompt, uint32_t h, uint32_t w, uint32_t slot = 0) override;
    // The velocity of the generated latents [h * w][24] at `sigma`, a level of the schedule, through that step's
    // head (the transformer runs them as rows of 2x2 patches).
    std::vector<float> step(std::span<const float> latents, float sigma, uint32_t slot = 0) override;
    // The checkpoint's grid (steps must be its six; no terminal stretch).
    std::vector<float> sigmas(uint32_t steps, uint32_t tokens, std::optional<float> terminal) const override;
    // Latents [h * w][24] to RGBA [16 h][16 w][4] in [-1, 1], at the 8-bit levels the released pipeline truncates
    // to.
    std::vector<float> decode(std::span<const float> latents, uint32_t h, uint32_t w) override;
    std::vector<float> encode(std::span<const float> rgba, uint32_t h, uint32_t w) override;
    uint32_t add_lora(const Lora& l) override;

    static constexpr uint32_t kMinArea = 2048 * 2048, kMaxTokens = 4096, kTextLayer = 50;
    static constexpr uint32_t kVisionShortEdge = 1024, kRefMinShortEdge = 1024, kRefMaxShortEdge = 2048;
    static constexpr float kRefT = 0.999f; // the level the reference rows are held at
    // The size a reference is resized to for one of its consumers: kept when its short edge is within [lo, hi],
    // otherwise scaled, keeping its ratio, so the short edge lands on the nearer bound, both sides rounded to
    // multiples of 32.
    static std::pair<uint32_t, uint32_t> bound_short_edge(uint32_t w, uint32_t h, uint32_t lo, uint32_t hi);
    // The text rows' modality tags (begin() reads them: the vision blocks' rows are image rows).
    void set_text_tags(std::vector<uint8_t> tags) { text_tags_ = std::move(tags); }
    const std::vector<uint8_t>& text_tags() const { return text_tags_; }
    // The transformer, loaded on first use (40 or 66 GB). Where it does not fit beside the text encoder's 49 GB
    // with room (1.35 x their files against the machine's memory, as the video engine judges it), the two take
    // turns: the text encoder is freed when the transformer loads and loaded again for the next prompt, the
    // transformer freed then.
    models::H3& transformer();
    bool keeps_both() const { return keeps_both_; }
    // Reference i's vision features, as set_references() encoded them.
    const models::QwenVision::Features& reference_features(size_t i) const { return refs_.at(i).features; }
    // The references' conditioning rows ([rows][96], noised to kRefT), as set_references() encoded them.
    std::span<const float> reference_rows() const { return cond_rows_; }
    models::H3Vae& vae() { return *vae_; }
    models::QwenVision& vision();
    // The references of the next generation: resized for the text encoder (their vision features) and the VAE
    // (their latents, sampled and noised to kRefT), in the request's order.
    void set_references(const Request& r) override;

protected:
    LoraTarget lora_target(const std::string&) override { return {}; }
    std::pair<uint32_t, uint32_t> canvas(uint32_t width, uint32_t height,
                                         std::optional<uint32_t> min_area) const override;

private:
    struct Ref {
        uint32_t lh = 0, lw = 0; // its latent grid (even)
        models::QwenVision::Features features;
    };
    std::vector<Ref> refs_;
    std::vector<float> cond_rows_; // the references' latent rows noised to kRefT, ahead of the generated rows
    std::vector<uint8_t> text_tags_;
    std::vector<float> grid_; // the schedule's sigmas, the terminal 0 last
    std::unique_ptr<models::H3> dit_;
    std::unique_ptr<models::H3Vae> vae_;
    std::unique_ptr<models::QwenVision> vision_;
    std::string text_dir_, dit_path_, heads_path_;
    std::vector<Lora> loras_; // merged when the transformer loads
    bool keeps_both_ = true;  // the text encoder stays loaded beside the transformer
};

} // namespace mako::image
