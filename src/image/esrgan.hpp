#pragma once

#include "image/conv.hpp"

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
// the original ESRGAN's layers), F32, F16 or BF16; the weights run as F16.
class Esrgan : public ConvRuntime {
public:
    // Throws std::runtime_error on a file it cannot use.
    Esrgan(const std::string& path, mtl::Context& ctx, gpu::Telemetry telemetry = gpu::Telemetry::off);
    const std::string& path() const { return path_; }
    // What a side is multiplied by: 2 for each upsampling.
    uint32_t factor() const { return 1u << ups_.size(); }
    uint32_t blocks() const { return uint32_t(body_.size()); }
    // RGBA [h][w][4] in [-1, 1] (alpha ignored) to RGBA [factor * h][factor * w][4] in [-1, 1] (alpha 1). An image
    // over `tile` pixels a side runs in tiles of that, each with `overlap` pixels of its neighbours as context; 0: all
    // at once. The scratch buffers (about 8 GB for a 1024x768 tile at 4x) are freed afterwards.
    std::vector<float> upscale(std::span<const float> rgba, uint32_t w, uint32_t h, uint32_t tile = 1024,
                               uint32_t overlap = 32);

private:
    struct Dense {
        Conv c[5];
    };
    struct Block {
        Dense d[3];
    };
    // The network over RGB [h][w][3] in [0, 1], into [factor * h][factor * w][3].
    void run(std::span<const float> rgb, uint32_t h, uint32_t w, float* out);

    std::string path_;
    Conv first_, trunk_, hr_, last_;
    std::vector<Block> body_;
    std::vector<Conv> ups_;
    mtl::Ref<MTL::Buffer> fifth_; // 0.2 for each of nf channels: the residuals' scale
    uint32_t nf_ = 0, gc_ = 0;
};

} // namespace mako::image
