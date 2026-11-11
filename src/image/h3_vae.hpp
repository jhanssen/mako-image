#pragma once

#include "models/conv.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace mako::image {

// MiniMax-H3-Image's VAE: the causal 3D CNN encoder of MiniMax-H3's video VAE over one frame (its last time taps:
// group norms, SiLU and reflect-padded 3x3 convolutions in six levels of two residual blocks, a stride-2 downsample
// after the first four, 16x down to 48 moments) and a 2D ViT decoder (a token per latent pixel plus four register
// tokens and a zero token, 24 blocks of 16 heads of 64 with two-axis rotary positions over half of each head, each
// token then a 16x16 patch of pixels). Both run in 256-pixel tiles whose overlaps are blended, as the model was
// released. The weights are F16 here (the ComfyUI repack's, or the release's F32 rounded).
class H3ImageVae final : public models::ConvRuntime {
public:
    // The weights (a safetensors file, or a directory holding diffusion_pytorch_model.safetensors) and the latent
    // statistics (a config.json with latents_mean / latents_std, the release's vae/config.json).
    H3ImageVae(const std::string& weights, const std::string& config, mtl::Context& ctx,
               gpu::Telemetry telemetry = gpu::Telemetry::off);

    const char* name() const override { return "image VAE"; }

    static constexpr uint32_t kLatent = 24, kPatch = 16, kDim = 1024, kHeads = 16, kHeadDim = 64, kRot = 32;
    static constexpr uint32_t kFfn = 4096, kRegisters = 4, kOut = 3 * kPatch * kPatch;
    static constexpr uint32_t kTile = 256, kTileOverlap = 64, kEncCh = 128;

    // Normalized latents [h][w][24] to pixels [16 h][16 w][3] in [0, 1].
    std::vector<float> decode(std::span<const float> latents, uint32_t h, uint32_t w);
    // An image [H][W][3] of 8-bit RGB (H and W multiples of 16) to its normalized latents [H / 16][W / 16][24]: the
    // posterior sampled with `noise` ([H / 16][W / 16][24]), the sample rounded to fp16 as the released pipeline does.
    // The encoder's weights load on first use.
    std::vector<float> encode(std::span<const uint8_t> rgb, uint32_t H, uint32_t W, std::span<const float> noise);
    // The posterior's mean and standard deviation, [H / 16][W / 16][24] each.
    void encode_moments(std::span<const uint8_t> rgb, uint32_t H, uint32_t W, std::vector<float>& mean,
                        std::vector<float>& sd);
    std::span<const float> latents_mean() const { return mean_; }
    std::span<const float> latents_std() const { return std_; }

private:
    struct Block {
        Tensor norm1, norm2, qkv, qkv_b, o, o_b, w1, w1_b, w2, w2_b, scale1, scale2;
    };
    // The ViT over a tile's latents [th][tw][24] (denormalized, after the post-quant projection): pixels
    // [16 th][16 tw][3] in the decoder's own (ImageNet-normalized) space.
    void vit(std::span<const float> z, uint32_t th, uint32_t tw, std::vector<float>& out);
    // Tiles over `length` pixels: each tile's start and the overlap with the one before it (0 for the first).
    struct Tile {
        uint32_t start, size, overlap;
    };
    static std::vector<Tile> tiles(uint32_t length);

    Tensor embed_, embed_b_, registers_, ones_, out_norm_w_, out_norm_b_, proj_out_, proj_out_b_;
    std::vector<Block> blocks_;
    std::vector<float> mean_, std_, pq_w_, pq_b_; // latent statistics; the post-quant 1x1 convolution
    std::string weights_;

    // The CNN encoder.
    struct EncRes {
        Tensor g1, b1, g2, b2;
        Conv c1, c2, sc;
        bool shortcut = false;
    };
    struct EncLevel {
        std::vector<EncRes> res;
        bool down = false;
        Conv down_conv;
    };
    struct Encoder {
        Conv in, out, quant;
        Tensor out_g, out_b;
        std::vector<EncLevel> levels;
    };
    std::unique_ptr<Encoder> enc_;
    void load_encoder();
    // One tile of ImageNet-normalized pixels [th][tw][3] to its moments [th / 16][tw / 16][48] (mean, log variance).
    void encode_tile(std::span<const float> pixels, uint32_t th, uint32_t tw, std::vector<float>& moments);
    // The whole image's moments, tiles blended.
    void moments(std::span<const uint8_t> rgb, uint32_t H, uint32_t W, std::vector<float>& out);
};

} // namespace mako::image
