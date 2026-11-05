#pragma once

#include "models/conv.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace mako::image {

// An ESRGAN upscaler (RRDBNet, as the original ESRGAN and Real-ESRGAN build it): a convolution from RGB to nf
// channels, residual-in-residual dense blocks (three dense blocks each of five 3x3 convolutions, every one reading
// the block's input and the earlier ones' outputs, gc channels each, LeakyReLU 0.2 between; residuals scaled by 0.2),
// a trunk convolution added to the first one's output, a nearest 2x upsampling and a convolution for each doubling,
// and two convolutions back to RGB. From a .safetensors in Real-ESRGAN's names (conv_first, body.N.rdbM.convK,
// conv_body, conv_upN, conv_hr, conv_last; tools/esrgan_to_safetensors.py converts the .pth checkpoints, renaming
// the original ESRGAN's layers), F32, F16 or BF16; the weights run as F16. The convolutions read padded half images
// through nine shifted views (conv3x3_dense in image.metal), without unfolding them.
class Esrgan : public models::ConvRuntime {
public:
    // Throws std::runtime_error on a file it cannot use.
    Esrgan(const std::string& path, mtl::Context& ctx, gpu::Telemetry telemetry = gpu::Telemetry::off);
    const std::string& path() const { return path_; }
    // What a side is multiplied by: 2 for each upsampling.
    uint32_t factor() const { return 1u << ups_.size(); }
    uint32_t blocks() const { return uint32_t(body_.size()); }
    // RGBA [h][w][4] in [-1, 1] (alpha ignored) to RGBA [factor * h][factor * w][4] in [-1, 1] (alpha 1). An image
    // over `tile` pixels a side runs in tiles of that, each with `overlap` pixels of its neighbours as context; 0: all
    // at once. The scratch buffers (about 11 GB for a 1024x768 tile at 4x) are freed afterwards.
    std::vector<float> upscale(std::span<const float> rgba, uint32_t w, uint32_t h, uint32_t tile = 1024,
                               uint32_t overlap = 32);

private:
    // A 3x3 convolution run without unfolding (the conv3x3_dense kernel): weights F16 [9][cout][k], tap-major, k the
    // input channels rounded up to 64 (zero weights past them).
    struct Direct {
        Tensor w, b;
        uint32_t cin = 0, cout = 0, k = 0;
    };
    struct Dense {
        Direct c[5];
    };
    struct Block {
        Dense d[3];
    };
    Direct load_direct(const safetensors::Files& files, const std::string& name);
    // y = conv(x) over a padded image of `rows` pixels, pw wide, x's rows ldx halves apart with `slack` zero rows
    // before and after it, y's ldy floats apart (the ring's rows garbage).
    void conv_direct(MTL::ComputeCommandEncoder*& enc, const Direct& c, MTL::Buffer* x, uint32_t ldx, uint32_t rows,
                     uint32_t pw, uint32_t slack, MTL::Buffer* y, uint32_t ldy);
    // The network over RGB [h][w][3] in [0, 1], into [factor * h][factor * w][3].
    void run(std::span<const float> rgb, uint32_t h, uint32_t w, float* out);

    std::string path_;
    Direct first_, trunk_, hr_, last_;
    std::vector<Block> body_;
    std::vector<Direct> ups_;
    mtl::Ref<MTL::Buffer> fifth_; // 0.2 for each of nf channels: the residuals' scale
    uint32_t nf_ = 0, gc_ = 0;
};

} // namespace mako::image
