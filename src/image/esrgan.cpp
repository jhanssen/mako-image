#include "image/esrgan.hpp"

#include <algorithm>
#include <cstring>
#include <format>
#include <stdexcept>

namespace mako::image {

namespace {

constexpr float kSlope = 0.2f;    // LeakyReLU's negative slope
constexpr float kResidual = 0.2f; // what a dense block's and a residual block's output is scaled by
constexpr uint32_t kTile = 128;   // pixel rows a threadgroup of the direct convolution

// The kernels' arguments (image.metal).
struct ConvArgs {
    uint32_t t, k, n, ldx, ldy, pw, slack;
};
struct PadArgs {
    uint32_t n, ldx, ldy, c0, pw, ph;
    float slope;
};
} // namespace

Esrgan::Esrgan(const std::string& path, mtl::Context& ctx, gpu::Telemetry telemetry)
    : ConvRuntime(ctx, telemetry), path_(path)
{
    auto files = safetensors::Files::open(path);
    if (!files)
        throw std::runtime_error(files.error());
    auto expect = [&](uint32_t cin, uint32_t cout, const std::string& name, uint32_t want_in, uint32_t want_out) {
        if (cin != want_in || cout != want_out)
            throw std::runtime_error(
                std::format("{}: {} is {} -> {} channels, not {} -> {}", path, name, cin, cout, want_in, want_out));
    };
    first_ = load_direct(*files, "conv_first");
    if (first_.cin != 3)
        throw std::runtime_error(std::format("{}: conv_first takes {} channels, not RGB", path, first_.cin));
    nf_ = first_.cout;
    for (uint32_t i = 0; files->find(std::format("body.{}.rdb1.conv1.weight", i)); ++i) {
        Block b;
        for (uint32_t d = 0; d < 3; ++d)
            for (uint32_t k = 0; k < 5; ++k)
                b.d[d].c[k] = load_direct(*files, std::format("body.{}.rdb{}.conv{}", i, d + 1, k + 1));
        body_.push_back(std::move(b));
    }
    if (body_.empty())
        throw std::runtime_error(path + ": no body.N.rdbM.convK blocks (not an ESRGAN in Real-ESRGAN's names)");
    gc_ = body_[0].d[0].c[0].cout;
    // Dense convolution k reads the block's nf channels and the k earlier outputs of gc, the last giving nf back.
    for (size_t i = 0; i < body_.size(); ++i)
        for (uint32_t d = 0; d < 3; ++d)
            for (uint32_t k = 0; k < 5; ++k) {
                const Direct& c = body_[i].d[d].c[k];
                expect(c.cin, c.cout, std::format("body.{}.rdb{}.conv{}", i, d + 1, k + 1), nf_ + k * gc_,
                       k == 4 ? nf_ : gc_);
            }
    trunk_ = load_direct(*files, "conv_body");
    expect(trunk_.cin, trunk_.cout, "conv_body", nf_, nf_);
    for (uint32_t i = 1; files->find(std::format("conv_up{}.weight", i)); ++i) {
        ups_.push_back(load_direct(*files, std::format("conv_up{}", i)));
        expect(ups_.back().cin, ups_.back().cout, std::format("conv_up{}", i), nf_, nf_);
    }
    hr_ = load_direct(*files, "conv_hr");
    expect(hr_.cin, hr_.cout, "conv_hr", nf_, nf_);
    last_ = load_direct(*files, "conv_last");
    expect(last_.cin, last_.cout, "conv_last", nf_, 3);
    fifth_ = mtl::Ref<MTL::Buffer>(ctx_.device()->newBuffer(uint64_t(nf_) * 4, MTL::ResourceStorageModeShared));
    std::fill_n(static_cast<float*>(fifth_->contents()), nf_, kResidual);
}

Esrgan::Direct Esrgan::load_direct(const safetensors::Files& files, const std::string& name)
{
    const auto r = files.find(name + ".weight");
    if (!r || !files.find(name + ".bias") || r.tensor->dims.size() != 4 || r.tensor->dims[0] != 3 ||
        r.tensor->dims[1] != 3 ||
        (r.tensor->type != gguf::Type::F32 && r.tensor->type != gguf::Type::F16 && r.tensor->type != gguf::Type::BF16))
        throw std::runtime_error(name + ": missing, or not an F32, F16 or BF16 3x3 convolution");
    Direct c;
    c.cin = uint32_t(r.tensor->dims[2]);
    c.cout = uint32_t(r.tensor->dims[3]);
    c.k = (c.cin + 63) / 64 * 64;
    c.w = reserve(uint64_t(9) * c.cout * c.k * 2, gguf::Type::F16, {c.k, c.cout, 9});
    c.b = reserve(uint64_t(c.cout) * 4, gguf::Type::F32, {c.cout});
    auto* dst = reinterpret_cast<__fp16*>(static_cast<char*>(c.w.buf->contents()) + c.w.off);
    std::fill_n(dst, uint64_t(9) * c.cout * c.k, __fp16(0));
    const auto v = read_floats(*r.file, *r.tensor); // [cout][cin][3][3]
    for (uint32_t o = 0; o < c.cout; ++o)
        for (uint32_t ch = 0; ch < c.cin; ++ch)
            for (uint32_t tap = 0; tap < 9; ++tap)
                dst[(uint64_t(tap) * c.cout + o) * c.k + ch] = __fp16(v[(uint64_t(o) * c.cin + ch) * 9 + tap]);
    const auto b = floats(files, name + ".bias");
    std::copy(b.begin(), b.end(), reinterpret_cast<float*>(static_cast<char*>(c.b.buf->contents()) + c.b.off));
    return c;
}

void Esrgan::conv_direct(MTL::ComputeCommandEncoder*& enc, const Direct& c, MTL::Buffer* x, uint32_t ldx,
                         uint32_t rows, uint32_t pw, uint32_t slack, MTL::Buffer* y, uint32_t ldy)
{
    const ConvArgs a{rows, c.k, c.cout, ldx, ldy, pw, slack};
    const uint32_t tn = c.cout <= 32 ? 32 : 64;
    enc->setComputePipelineState(pso(std::format("conv3x3_{}x{}", kTile, tn)));
    enc->setBuffer(c.w.buf, c.w.off, 0);
    enc->setBuffer(x, 0, 1);
    enc->setBuffer(y, 0, 2);
    enc->setBytes(&a, sizeof(a), 3);
    enc->dispatchThreadgroups(MTL::Size((c.cout + tn - 1) / tn, (rows + kTile - 1) / kTile, 1), MTL::Size(128, 1, 1));
    after(enc, "conv3x3_dense");
}

void Esrgan::run(std::span<const float> rgb, uint32_t h, uint32_t w, float* out)
{
    const uint64_t rows = uint64_t(h) * w, f = factor();
    // The dense stage runs on the padded image: a zero ring around it, and `slack` zero rows before and after it in
    // the half buffers the convolutions read through shifted views.
    const uint32_t pw = w + 2, ph = h + 2, P = pw * ph, slack = pw + 1 + kTile, wide = nf_ + 4 * gc_;
    const uint64_t prows = uint64_t(P) + 2 * slack;
    MTL::Buffer* in = scratch("e_in", prows * 64 * 2); // RGB in the first of 64 half channels
    MTL::Buffer* d = scratch("e_d", prows * wide * 2); // a dense block's input and its first four outputs
    MTL::Buffer* fea = scratch("e_fea", uint64_t(P) * nf_ * 4);
    MTL::Buffer* x = scratch("e_x", uint64_t(P) * nf_ * 4);
    MTL::Buffer* r = scratch("e_r", uint64_t(P) * nf_ * 4);
    MTL::Buffer* e = scratch("e_e", uint64_t(P) * nf_ * 4);
    MTL::Buffer* y32 = scratch("e_y", uint64_t(P) * gc_ * 4);
    // The upsampling stage: padded half images at each size, with their slack, and one float output at the largest.
    struct Level {
        uint32_t w, h, pw, ph, P, slack;
        uint64_t prows;
        MTL::Buffer *u = nullptr, *v = nullptr; // two padded half images
    };
    std::vector<Level> levels;
    for (uint32_t lv = 0, lw = w, lh = h; lv <= ups_.size(); ++lv, lw *= 2, lh *= 2) {
        Level L{lw, lh, lw + 2, lh + 2, (lw + 2) * (lh + 2), lw + 3 + kTile, 0};
        L.prows = uint64_t(L.P) + 2 * L.slack;
        L.u = scratch(std::format("e_u{}", lv), L.prows * nf_ * 2);
        if (lv > 0)
            L.v = scratch(std::format("e_v{}", lv), L.prows * nf_ * 2);
        levels.push_back(L);
    }
    const Level& top = levels.back();
    MTL::Buffer* y = scratch("e_yup", uint64_t(top.P) * nf_ * 4);
    MTL::Buffer* res = scratch("e_out", rows * f * f * 3 * 4);
    {
        auto* ip = static_cast<__fp16*>(in->contents());
        std::memset(ip, 0, prows * 64 * 2);
        for (uint32_t y = 0; y < h; ++y)
            for (uint32_t xx = 0; xx < w; ++xx) {
                __fp16* px = ip + (uint64_t(slack) + uint64_t(y + 1) * pw + xx + 1) * 64;
                for (uint32_t ch = 0; ch < 3; ++ch)
                    px[ch] = __fp16(rgb[(uint64_t(y) * w + xx) * 3 + ch]);
            }
    }
    auto copy = [&](MTL::ComputeCommandEncoder*& enc, MTL::Buffer* from, MTL::Buffer* to) {
        const uint32_t n = uint32_t(uint64_t(P) * nf_);
        dispatch(enc, "copy_f32", {from, to}, &n, sizeof(n), MTL::Size(n, 1, 1), MTL::Size(256, 1, 1));
    };
    auto add_fifth = [&](MTL::ComputeCommandEncoder*& enc, MTL::Buffer* into, MTL::Buffer* what) { // into += 0.2 what
        dispatch(enc, "scale_add", {into, what, fifth_.get()}, &nf_, sizeof(nf_), MTL::Size(nf_, P, 1),
                 MTL::Size(64, 1, 1));
    };
    // The bias (and LeakyReLU) of a padded float image of n channels in place, its ring zeroed.
    auto bias_pad = [&](MTL::ComputeCommandEncoder*& enc, MTL::Buffer* y, const Tensor& bias, uint32_t n,
                        uint32_t rows_, uint32_t pw_, uint32_t ph_, float slope) {
        const PadArgs pa{n, n, n, 0, pw_, ph_, slope};
        enc->setComputePipelineState(pso("bias_pad"));
        enc->setBuffer(y, 0, 0);
        enc->setBuffer(bias.buf, bias.off, 1);
        enc->setBytes(&pa, sizeof(pa), 2);
        enc->dispatchThreads(MTL::Size(n, rows_, 1), MTL::Size(64, 1, 1));
        after(enc, "bias_pad");
    };
    // The same from a float image into a padded half one (columns from c0 of rows ldy wide, at its slack).
    auto bias_pad_half = [&](MTL::ComputeCommandEncoder*& enc, MTL::Buffer* from, const Tensor& bias, uint32_t n,
                             MTL::Buffer* to, uint64_t to_off, uint32_t ldy, uint32_t c0, uint32_t rows_, uint32_t pw_,
                             uint32_t ph_, float slope) {
        const PadArgs pa{n, n, ldy, c0, pw_, ph_, slope};
        enc->setComputePipelineState(pso("bias_pad_half"));
        enc->setBuffer(from, 0, 0);
        enc->setBuffer(bias.buf, bias.off, 1);
        enc->setBuffer(to, to_off, 2);
        enc->setBytes(&pa, sizeof(pa), 3);
        enc->dispatchThreads(MTL::Size(n, rows_, 1), MTL::Size(64, 1, 1));
        after(enc, "bias_pad_half");
    };
    // nf float channels of a padded image into a half one's first columns (rows ldy wide, at its slack).
    auto to_half = [&](MTL::ComputeCommandEncoder*& enc, MTL::Buffer* from, MTL::Buffer* to, uint64_t to_off,
                       uint32_t ldy) {
        const PadArgs pa{nf_, nf_, ldy, 0, pw, ph, 1.0f};
        enc->setComputePipelineState(pso("to_half_cols"));
        enc->setBuffer(from, 0, 0);
        enc->setBuffer(to, to_off, 1);
        enc->setBytes(&pa, sizeof(pa), 2);
        enc->dispatchThreads(MTL::Size(nf_, P, 1), MTL::Size(64, 1, 1));
        after(enc, "to_half_cols");
    };
    auto into_d = [&](MTL::ComputeCommandEncoder*& enc, MTL::Buffer* from) {
        to_half(enc, from, d, uint64_t(slack) * wide * 2, wide);
    };
    auto zero = [&](MTL::ComputeCommandEncoder*& enc, MTL::Buffer* buf, uint64_t bytes) {
        const uint32_t n = uint32_t(bytes / 16);
        dispatch(enc, "zero_half8", {buf}, &n, sizeof(n), MTL::Size(n, 1, 1), MTL::Size(256, 1, 1));
    };

    MTL::CommandBuffer* cb = ctx_.queue()->commandBuffer();
    cur_cb_ = cb;
    MTL::ComputeCommandEncoder* enc = encoder(cb, 0, -1);
    zero(enc, d, prows * wide * 2);
    for (const Level& L : levels) {
        zero(enc, L.u, L.prows * nf_ * 2);
        if (L.v)
            zero(enc, L.v, L.prows * nf_ * 2);
    }
    conv_direct(enc, first_, in, 64, P, pw, slack, fea, nf_);
    bias_pad(enc, fea, first_.b, nf_, P, pw, ph, 1.0f);
    probe(enc, "first", -1, fea, uint64_t(P) * nf_);
    copy(enc, fea, x);
    for (size_t i = 0; i < body_.size(); ++i) {
        copy(enc, x, r); // the block's input, for its residual
        for (const Dense& dn : body_[i].d) {
            into_d(enc, x);
            for (uint32_t k = 0; k < 4; ++k) {
                conv_direct(enc, dn.c[k], d, wide, P, pw, slack, y32, gc_);
                bias_pad_half(enc, y32, dn.c[k].b, gc_, d, uint64_t(slack) * wide * 2, wide, nf_ + k * gc_, P, pw, ph,
                              kSlope);
            }
            conv_direct(enc, dn.c[4], d, wide, P, pw, slack, e, nf_);
            bias_pad(enc, e, dn.c[4].b, nf_, P, pw, ph, 1.0f);
            add_fifth(enc, x, e);
        }
        add_fifth(enc, r, x);
        std::swap(x, r);
        if (i == 0)
            probe(enc, "b0", -1, x, uint64_t(P) * nf_);
        if (i % 4 == 3) // the GPU starts on the blocks so far while the rest are encoded
            commit_ahead(cb, enc);
    }
    into_d(enc, x);
    conv_direct(enc, trunk_, d, wide, P, pw, slack, e, nf_);
    bias_pad(enc, e, trunk_.b, nf_, P, pw, ph, 1.0f);
    probe(enc, "trunk", -1, e, uint64_t(P) * nf_);
    {
        const uint32_t n = uint32_t(uint64_t(P) * nf_);
        dispatch(enc, "add", {fea, e, fea}, &n, sizeof(n), MTL::Size(n, 1, 1), MTL::Size(256, 1, 1));
    }
    probe(enc, "fea", -1, fea, uint64_t(P) * nf_);
    // Up: each level's image doubled by nearest into the next level's u, convolved into its v.
    to_half(enc, fea, levels[0].u, uint64_t(levels[0].slack) * nf_ * 2, nf_);
    MTL::Buffer* cur = levels[0].u;
    uint64_t cur_off = uint64_t(levels[0].slack) * nf_ * 2;
    for (size_t lv = 0; lv < ups_.size(); ++lv) {
        const Level &from = levels[lv], &to = levels[lv + 1];
        const PadArgs pa{nf_, nf_, nf_, 0, from.pw, from.ph, 1.0f};
        enc->setComputePipelineState(pso("upsample2x_pad"));
        enc->setBuffer(cur, cur_off, 0);
        enc->setBuffer(to.u, uint64_t(to.slack) * nf_ * 2, 1);
        enc->setBytes(&pa, sizeof(pa), 2);
        enc->dispatchThreads(MTL::Size(nf_, to.w, to.h), MTL::Size(64, 1, 1));
        after(enc, "upsample2x_pad");
        conv_direct(enc, ups_[lv], to.u, nf_, to.P, to.pw, to.slack, y, nf_);
        bias_pad_half(enc, y, ups_[lv].b, nf_, to.v, uint64_t(to.slack) * nf_ * 2, nf_, 0, to.P, to.pw, to.ph, kSlope);
        cur = to.v;
        cur_off = uint64_t(to.slack) * nf_ * 2;
    }
    conv_direct(enc, hr_, cur, nf_, top.P, top.pw, top.slack, y, nf_);
    bias_pad_half(enc, y, hr_.b, nf_, top.u, uint64_t(top.slack) * nf_ * 2, nf_, 0, top.P, top.pw, top.ph, kSlope);
    conv_direct(enc, last_, top.u, nf_, top.P, top.pw, top.slack, y, 3);
    bias_pad(enc, y, last_.b, 3, top.P, top.pw, top.ph, 1.0f);
    {
        const PadArgs pa{3, 3, 3, 0, top.pw, top.ph, 1.0f};
        dispatch(enc, "unpad", {y, res}, &pa, sizeof(pa), MTL::Size(3, top.w, top.h), MTL::Size(64, 1, 1));
    }
    probe(enc, "out", -1, res, uint64_t(top.h) * top.w * 3);
    enc->endEncoding();
    finish(cb);
    std::memcpy(out, res->contents(), rows * f * f * 3 * 4);
}

std::vector<float> Esrgan::upscale(std::span<const float> rgba, uint32_t w, uint32_t h, uint32_t tile,
                                   uint32_t overlap)
{
    if (rgba.size() != uint64_t(w) * h * 4)
        throw std::runtime_error("upscale: the image must be RGBA [h][w][4]");
    const uint32_t f = factor(), W = w * f;
    std::vector<float> out(uint64_t(W) * h * f * 4, 1.0f);
    const uint32_t step = tile && tile < std::max(w, h) ? tile : std::max(w, h);
    for (uint32_t y0 = 0; y0 < h; y0 += step)
        for (uint32_t x0 = 0; x0 < w; x0 += step) {
            const uint32_t y1 = std::min(h, y0 + step), x1 = std::min(w, x0 + step);
            // The tile with its context: what the network sees.
            const uint32_t cy0 = y0 > overlap ? y0 - overlap : 0, cx0 = x0 > overlap ? x0 - overlap : 0;
            const uint32_t cy1 = std::min(h, y1 + overlap), cx1 = std::min(w, x1 + overlap);
            const uint32_t th = cy1 - cy0, tw = cx1 - cx0;
            std::vector<float> rgb(uint64_t(th) * tw * 3);
            for (uint32_t y = 0; y < th; ++y)
                for (uint32_t x = 0; x < tw; ++x)
                    for (uint32_t ch = 0; ch < 3; ++ch)
                        rgb[(uint64_t(y) * tw + x) * 3 + ch] =
                            (rgba[(uint64_t(cy0 + y) * w + cx0 + x) * 4 + ch] + 1.0f) * 0.5f;
            std::vector<float> res(uint64_t(th) * tw * f * f * 3);
            run(rgb, th, tw, res.data());
            for (uint32_t y = y0 * f; y < y1 * f; ++y)
                for (uint32_t x = x0 * f; x < x1 * f; ++x)
                    for (uint32_t ch = 0; ch < 3; ++ch)
                        out[(uint64_t(y) * W + x) * 4 + ch] = std::clamp(
                            res[(uint64_t(y - cy0 * f) * tw * f + x - cx0 * f) * 3 + ch] * 2.0f - 1.0f, -1.0f, 1.0f);
        }
    scratch_.clear();
    return out;
}

} // namespace mako::image
