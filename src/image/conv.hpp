#pragma once

#include "gpu/runtime.hpp"
#include "safetensors/safetensors.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace mako::image {

// A tensor's values as floats (F32, F16 or BF16).
std::vector<float> read_floats(const safetensors::File& f, const gguf::Tensor& t);

// What the image models' convolutions share: 3x3 and 1x1 convolutions as GEMMs over channels-last images
// ([height][width][channels], a pixel a row) with F16 weights, and their loading. Engine (the VAEs) derives from it;
// Esrgan too, for the weight helpers (its own convolutions are in esrgan.cpp).
class ConvRuntime : public gpu::Runtime {
protected:
    ConvRuntime(mtl::Context& ctx, gpu::Telemetry telemetry) : gpu::Runtime(ctx, telemetry) {}

    // Weights F16 [cout][kpad] (cin * k * k columns, a 3x3's zero padded to a multiple of 64). A causal 3D
    // convolution's weights are its last time tap's (a single frame).
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
    // y[r][i] += b[i] over rows of n.
    void bias(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* y, const Tensor& b, uint32_t n, uint32_t rows,
              uint64_t y_off = 0);
    // A tensor's values as floats.
    static std::vector<float> floats(const safetensors::Files& files, const std::string& name);
};

} // namespace mako::image
