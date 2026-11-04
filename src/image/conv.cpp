#include "image/conv.hpp"

#include "gguf/read_pool.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace mako::image {

namespace {

struct Im2colArgs {
    uint32_t h, w, c, p0, rows, kpad;
};

uint32_t round_up(uint32_t x, uint32_t m) { return (x + m - 1) / m * m; }

} // namespace

std::vector<float> read_floats(const safetensors::File& f, const gguf::Tensor& t)
{
    const uint64_t n = t.elements();
    std::vector<float> out(n);
    std::vector<std::byte> raw(t.bytes);
    gguf::read_exact(f.fd(), raw.data(), t.bytes, t.offset);
    for (uint64_t i = 0; i < n; ++i) {
        if (t.type == gguf::Type::F32) {
            std::memcpy(&out[i], raw.data() + i * 4, 4);
        } else if (t.type == gguf::Type::F16) {
            __fp16 h;
            std::memcpy(&h, raw.data() + i * 2, 2);
            out[i] = float(h);
        } else {
            uint16_t b;
            std::memcpy(&b, raw.data() + i * 2, 2);
            const uint32_t u = uint32_t(b) << 16;
            std::memcpy(&out[i], &u, 4);
        }
    }
    return out;
}

std::vector<float> ConvRuntime::floats(const safetensors::Files& files, const std::string& name)
{
    const auto r = files.find(name);
    if (!r)
        throw std::runtime_error("missing tensor " + name);
    return read_floats(*r.file, *r.tensor);
}

void ConvRuntime::bias(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* y, const Tensor& b, uint32_t n, uint32_t rows,
                       uint64_t y_off)
{
    enc->setComputePipelineState(pso("bias_rows"));
    enc->setBuffer(y, y_off, 0);
    enc->setBuffer(b.buf, b.off, 1);
    enc->setBytes(&n, sizeof(n), 2);
    enc->dispatchThreads(MTL::Size(n, rows, 1), MTL::Size(64, 1, 1));
    after(enc, "bias_rows");
}

ConvRuntime::Conv ConvRuntime::load_conv(const safetensors::Files& files, std::span<const std::string> names,
                                         uint32_t k)
{
    struct Part {
        safetensors::Files::Ref r;
        uint32_t cout, kt;
    };
    std::vector<Part> parts;
    Conv c;
    c.k = k;
    for (const std::string& name : names) {
        const auto r = files.find(name + ".weight");
        const size_t nd = r ? r.tensor->dims.size() : 0;
        if (!r || !files.find(name + ".bias") ||
            (r.tensor->type != gguf::Type::F32 && r.tensor->type != gguf::Type::BF16 &&
             r.tensor->type != gguf::Type::F16) ||
            (nd != 2 && nd != 4 && nd != 5))
            throw std::runtime_error(name + ": missing, or not an F32, F16 or BF16 convolution");
        // A linear layer is a 1x1 convolution.
        if (nd == 2 ? k != 1 : r.tensor->dims[0] != k || r.tensor->dims[1] != k)
            throw std::runtime_error(name + ": unexpected kernel size");
        const uint32_t cin = uint32_t(r.tensor->dims[nd - 2]);
        if (c.cin && cin != c.cin)
            throw std::runtime_error(name + ": input channels differ from the others'");
        c.cin = cin;
        // A causal 3D convolution over one frame sees only its last time tap (the earlier ones read the zero padding).
        parts.push_back({r, uint32_t(r.tensor->dims[nd - 1]), nd == 5 ? uint32_t(r.tensor->dims[2]) : 1});
        c.cout += parts.back().cout;
    }
    const uint32_t cols = c.cin * k * k, kk = k * k;
    // 3x3: the unfolded rows padded for the direct GEMM; 1x1: the input's own rows (the staged GEMM where they are not
    // a multiple of 64).
    c.kpad = round_up(cols, k == 3 ? 64 : 32);
    // Half, as the activations it multiplies are (the direct GEMM reads it unstaged).
    c.w = reserve(uint64_t(c.cout) * c.kpad * 2, gguf::Type::F16, {c.kpad, c.cout});
    c.b = reserve(uint64_t(c.cout) * 4, gguf::Type::F32, {c.cout});
    auto* dst = reinterpret_cast<__fp16*>(static_cast<char*>(c.w.buf->contents()) + c.w.off);
    auto* bias = reinterpret_cast<float*>(static_cast<char*>(c.b.buf->contents()) + c.b.off);
    uint32_t o0 = 0;
    for (size_t pi = 0; pi < parts.size(); ++pi) {
        const Part& p = parts[pi];
        std::vector<float> row(uint64_t(cols) * p.kt);
        const gguf::Type type = p.r.tensor->type;
        std::vector<uint16_t> raw(type != gguf::Type::F32 ? row.size() : 0);
        for (uint32_t o = 0; o < p.cout; ++o) {
            if (raw.empty()) {
                gguf::read_exact(p.r.file->fd(), row.data(), row.size() * 4,
                                 p.r.tensor->offset + uint64_t(o) * row.size() * 4);
            } else {
                gguf::read_exact(p.r.file->fd(), raw.data(), raw.size() * 2,
                                 p.r.tensor->offset + uint64_t(o) * raw.size() * 2);
                for (size_t i = 0; i < raw.size(); ++i) {
                    if (type == gguf::Type::F16) {
                        __fp16 h;
                        std::memcpy(&h, &raw[i], 2);
                        row[i] = float(h);
                    } else {
                        const uint32_t bits = uint32_t(raw[i]) << 16;
                        std::memcpy(&row[i], &bits, 4);
                    }
                }
            }
            for (uint32_t i = 0; i < c.kpad; ++i) {
                float v = 0.0f;
                if (i < cols)
                    v = row[(uint64_t(i / kk) * p.kt + p.kt - 1) * kk + i % kk];
                dst[uint64_t(o0 + o) * c.kpad + i] = __fp16(v);
            }
        }
        const auto b = floats(files, names[pi] + ".bias");
        std::copy(b.begin(), b.end(), bias + o0);
        o0 += p.cout;
    }
    return c;
}

void ConvRuntime::conv(MTL::ComputeCommandEncoder*& enc, const Conv& c, MTL::Buffer* x, uint32_t h, uint32_t w,
                       MTL::Buffer* y)
{
    if (c.stride == 2) { // a 3x3 to the half-size image, padded by c.pad above and left, 1 below and right
        const uint32_t oh = (h + c.pad - 2) / 2 + 1, ow = (w + c.pad - 2) / 2 + 1, rows = oh * ow;
        const uint32_t chunk = std::max(256u, uint32_t((256ull << 20) / (uint64_t(c.kpad) * 2)) / 256 * 256);
        MTL::Buffer* col = scratch("v_col", uint64_t(std::min(chunk, rows)) * c.kpad * 2);
        for (uint32_t p0 = 0; p0 < rows; p0 += chunk) {
            const uint32_t n = std::min(chunk, rows - p0);
            const uint32_t a[8] = {h, w, c.cin, p0, n, c.kpad, ow, c.pad};
            dispatch(enc, "im2col3x3_s2", {x, col}, a, sizeof(a), MTL::Size(c.kpad, n, 1), MTL::Size(256, 1, 1));
            gemm_->encode(enc, gguf::Type::F16, c.kpad, c.cout, n, c.w.buf, c.w.off, col, 0, y,
                          uint64_t(p0) * c.cout * 4, 0, 0, true);
            after(enc, "conv3x3_s2");
        }
        bias(enc, y, c.b, c.cout, rows);
        return;
    }
    const uint32_t rows = h * w;
    if (c.k == 1) {
        MTL::Buffer* in = x;
        if (c.kpad != c.cin) { // rows zero padded to the GEMM's
            in = scratch("v_pad", uint64_t(rows) * c.kpad * 4);
            std::memset(in->contents(), 0, uint64_t(rows) * c.kpad * 4);
            const uint32_t a[2] = {c.cin, c.kpad};
            dispatch(enc, "copy_rows", {x, in}, a, sizeof(a), MTL::Size(c.cin, rows, 1), MTL::Size(256, 1, 1));
        }
        gemm_->encode(enc, gguf::Type::F16, c.kpad, c.cout, rows, c.w.buf, c.w.off,
                      halves(enc, in, 0, rows, c.kpad, 0), 0, y, 0, 0, 0, c.kpad % 64 == 0);
        after(enc, "conv1x1");
    } else {
        // Image rows a pass, so the unfolded copy stays near 256 MB, in multiples of 256 (the GEMM's largest tiles).
        const uint32_t chunk = std::max(256u, uint32_t((256ull << 20) / (uint64_t(c.kpad) * 2)) / 256 * 256);
        MTL::Buffer* col = scratch("v_col", uint64_t(std::min(chunk, rows)) * c.kpad * 2);
        for (uint32_t p0 = 0; p0 < rows; p0 += chunk) {
            const uint32_t n = std::min(chunk, rows - p0);
            const Im2colArgs a{h, w, c.cin, p0, n, c.kpad};
            dispatch(enc, "im2col3x3", {x, col}, &a, sizeof(a), MTL::Size(c.kpad, n, 1), MTL::Size(256, 1, 1));
            gemm_->encode(enc, gguf::Type::F16, c.kpad, c.cout, n, c.w.buf, c.w.off, col, 0, y,
                          uint64_t(p0) * c.cout * 4, 0, 0, true);
            after(enc, "conv3x3");
        }
    }
    bias(enc, y, c.b, c.cout, rows);
}

} // namespace mako::image
