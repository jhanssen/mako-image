#include "image/esrgan.hpp"

#include <algorithm>
#include <cstring>
#include <format>
#include <stdexcept>

namespace mako::image {

namespace {

constexpr float kSlope = 0.2f;    // LeakyReLU's negative slope
constexpr float kResidual = 0.2f; // what a dense block's and a residual block's output is scaled by

// The upsample2x kernel's arguments (the duplicating shortcut's fields unused).
struct UpArgs {
    uint32_t h, w, c, c_out, ft, repeats;
};

} // namespace

Esrgan::Esrgan(const std::string& path, mtl::Context& ctx, gpu::Telemetry telemetry)
    : ConvRuntime(ctx, telemetry), path_(path)
{
    auto files = safetensors::Files::open(path);
    if (!files)
        throw std::runtime_error(files.error());
    auto one = [&](const std::string& name) { return load_conv(*files, std::span(&name, 1), 3); };
    auto expect = [&](const Conv& c, const std::string& name, uint32_t cin, uint32_t cout) {
        if (c.cin != cin || c.cout != cout)
            throw std::runtime_error(
                std::format("{}: {} is {} -> {} channels, not {} -> {}", path, name, c.cin, c.cout, cin, cout));
    };
    first_ = one("conv_first");
    if (first_.cin != 3)
        throw std::runtime_error(std::format("{}: conv_first takes {} channels, not RGB", path, first_.cin));
    nf_ = first_.cout;
    for (uint32_t i = 0; files->find(std::format("body.{}.rdb1.conv1.weight", i)); ++i) {
        Block b;
        for (uint32_t d = 0; d < 3; ++d)
            for (uint32_t k = 0; k < 5; ++k)
                b.d[d].c[k] = one(std::format("body.{}.rdb{}.conv{}", i, d + 1, k + 1));
        body_.push_back(std::move(b));
    }
    if (body_.empty())
        throw std::runtime_error(path + ": no body.N.rdbM.convK blocks (not an ESRGAN in Real-ESRGAN's names)");
    gc_ = body_[0].d[0].c[0].cout;
    // Dense convolution k reads the block's nf channels and the k earlier outputs of gc, the last giving nf back.
    for (size_t i = 0; i < body_.size(); ++i)
        for (uint32_t d = 0; d < 3; ++d)
            for (uint32_t k = 0; k < 5; ++k)
                expect(body_[i].d[d].c[k], std::format("body.{}.rdb{}.conv{}", i, d + 1, k + 1), nf_ + k * gc_,
                       k == 4 ? nf_ : gc_);
    trunk_ = one("conv_body");
    expect(trunk_, "conv_body", nf_, nf_);
    for (uint32_t i = 1; files->find(std::format("conv_up{}.weight", i)); ++i) {
        ups_.push_back(one(std::format("conv_up{}", i)));
        expect(ups_.back(), std::format("conv_up{}", i), nf_, nf_);
    }
    hr_ = one("conv_hr");
    expect(hr_, "conv_hr", nf_, nf_);
    last_ = one("conv_last");
    expect(last_, "conv_last", nf_, 3);
    fifth_ = mtl::Ref<MTL::Buffer>(ctx_.device()->newBuffer(uint64_t(nf_) * 4, MTL::ResourceStorageModeShared));
    std::fill_n(static_cast<float*>(fifth_->contents()), nf_, kResidual);
}

void Esrgan::run(std::span<const float> rgb, uint32_t h, uint32_t w, float* out)
{
    const uint64_t rows = uint64_t(h) * w, f = factor();
    const uint32_t wide = nf_ + 4 * gc_; // a dense block's input and its four first outputs side by side
    MTL::Buffer* in = scratch("e_in", rows * 3 * 4);
    std::memcpy(in->contents(), rgb.data(), rows * 3 * 4);
    MTL::Buffer* fea = scratch("e_fea", rows * nf_ * 4);
    MTL::Buffer* x = scratch("e_x", rows * nf_ * 4);
    MTL::Buffer* r = scratch("e_r", rows * nf_ * 4);
    MTL::Buffer* e = scratch("e_e", rows * nf_ * 4);
    MTL::Buffer* d = scratch("e_d", rows * wide * 4);
    // The upsampled stages, at the output's size.
    MTL::Buffer* a = scratch("e_a", rows * f * f * nf_ * 4);
    MTL::Buffer* b = scratch("e_b", rows * f * f * nf_ * 4);
    MTL::Buffer* res = scratch("e_out", rows * f * f * 3 * 4);
    auto copy = [&](MTL::ComputeCommandEncoder*& enc, MTL::Buffer* from, MTL::Buffer* to) {
        const uint32_t n = uint32_t(rows * nf_);
        dispatch(enc, "copy_f32", {from, to}, &n, sizeof(n), MTL::Size(n, 1, 1), MTL::Size(256, 1, 1));
    };
    auto add_fifth = [&](MTL::ComputeCommandEncoder*& enc, MTL::Buffer* into, MTL::Buffer* what) { // into += 0.2 what
        dispatch(enc, "scale_add", {into, what, fifth_.get()}, &nf_, sizeof(nf_), MTL::Size(nf_, rows, 1),
                 MTL::Size(64, 1, 1));
    };

    MTL::CommandBuffer* cb = ctx_.queue()->commandBuffer();
    cur_cb_ = cb;
    MTL::ComputeCommandEncoder* enc = encoder(cb, 0, -1);
    conv(enc, first_, in, h, w, fea);
    probe(enc, "first", -1, fea, rows * nf_);
    copy(enc, fea, x);
    for (size_t i = 0; i < body_.size(); ++i) {
        copy(enc, x, r); // the block's input, for its residual
        for (const Dense& dn : body_[i].d) {
            const uint32_t cw[2] = {nf_, wide};
            dispatch(enc, "copy_rows", {x, d}, cw, sizeof(cw), MTL::Size(nf_, rows, 1), MTL::Size(256, 1, 1));
            for (uint32_t k = 0; k < 4; ++k)
                conv(enc, dn.c[k], d, h, w, d, {wide, wide, uint64_t(nf_ + k * gc_) * 4, kSlope});
            if (i == 0 && &dn == body_[0].d)
                probe(enc, "b0_rdb1_d", -1, d, rows * wide);
            conv(enc, dn.c[4], d, h, w, e, {wide, 0, 0, 1.0f});
            if (i == 0 && &dn == body_[0].d)
                probe(enc, "b0_rdb1_e", -1, e, rows * nf_);
            add_fifth(enc, x, e);
        }
        add_fifth(enc, r, x);
        std::swap(x, r);
        if (i == 0)
            probe(enc, "b0", -1, x, rows * nf_);
        if (i % 4 == 3) // the GPU starts on the blocks so far while the rest are encoded
            commit_ahead(cb, enc);
    }
    conv(enc, trunk_, x, h, w, e);
    probe(enc, "trunk", -1, e, rows * nf_);
    {
        const uint32_t n = uint32_t(rows * nf_);
        dispatch(enc, "add", {fea, e, fea}, &n, sizeof(n), MTL::Size(n, 1, 1), MTL::Size(256, 1, 1));
    }
    probe(enc, "fea", -1, fea, rows * nf_);
    MTL::Buffer* cur = fea;
    uint32_t hh = h, ww = w;
    for (const Conv& up : ups_) {
        const UpArgs ua{hh, ww, nf_, nf_, 1, 1};
        dispatch(enc, "upsample2x", {cur, a}, &ua, sizeof(ua), MTL::Size(nf_, 2 * ww, 2 * hh), MTL::Size(64, 1, 1));
        hh *= 2;
        ww *= 2;
        conv(enc, up, a, hh, ww, b, {0, 0, 0, kSlope});
        probe(enc, std::format("up{}", &up - ups_.data() + 1), -1, b, uint64_t(hh) * ww * nf_);
        cur = b;
    }
    conv(enc, hr_, cur, hh, ww, a, {0, 0, 0, kSlope});
    probe(enc, "hr", -1, a, uint64_t(hh) * ww * nf_);
    conv(enc, last_, a, hh, ww, res);
    probe(enc, "out", -1, res, uint64_t(hh) * ww * 3);
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
