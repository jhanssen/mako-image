#include "image/h3_vae.hpp"

#include "log/log.hpp"

#include <glaze/glaze.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <numbers>
#include <stdexcept>

namespace mako::image {

namespace {

struct LnArgs {
    uint32_t n;
    float eps;
};

constexpr float kEps = 1e-5f, kRopeTheta = 100.0f;
constexpr uint32_t kEmbedCols = 32; // the 24 latent channels padded for the GEMM
constexpr float kPixelMean[3] = {0.485f, 0.456f, 0.406f}, kPixelStd[3] = {0.229f, 0.224f, 0.225f};

safetensors::Files open_files(const std::string& path)
{
    auto f = safetensors::Files::open(path);
    if (!f)
        throw std::runtime_error(f.error());
    return std::move(*f);
}

} // namespace

H3ImageVae::H3ImageVae(const std::string& weights, const std::string& config, mtl::Context& ctx,
                       gpu::Telemetry telemetry)
    : ConvRuntime(ctx, telemetry),
      weights_(std::filesystem::is_directory(weights) ? weights + "/diffusion_pytorch_model.safetensors" : weights)
{
    {
        std::ifstream in(config, std::ios::binary);
        const std::string text{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
        glz::generic j;
        if (!in || glz::read_json(j, text) || !j.contains("latents_mean") || !j.contains("latents_std"))
            throw std::runtime_error(config + ": no latents_mean / latents_std");
        for (const auto& v : j["latents_mean"].get_array())
            mean_.push_back(float(v.get_number()));
        for (const auto& v : j["latents_std"].get_array())
            std_.push_back(float(v.get_number()));
        if (mean_.size() != kLatent || std_.size() != kLatent)
            throw std::runtime_error(config + ": latent statistics are not over 24 channels");
    }
    const auto f = open_files(weights_);
    if (!f.find("decoder.x_embedder.weight") || !f.find("encoder.conv_in.weight"))
        throw std::runtime_error(weights_ + ": not MiniMax-H3-Image's VAE");
    pq_w_ = floats(f, "post_quant_conv.weight"); // [24][24][1][1]
    pq_b_ = floats(f, "post_quant_conv.bias");
    if (pq_w_.size() != kLatent * kLatent || pq_b_.size() != kLatent)
        throw std::runtime_error("image VAE: the post-quant convolution is not 24 -> 24");
    // The latent embedding with its input padded to 32 columns.
    {
        const auto w = floats(f, "decoder.x_embedder.weight"); // [1024][24]
        if (w.size() != uint64_t(kDim) * kLatent)
            throw std::runtime_error("image VAE: the embedding is not 24 -> 1024");
        embed_ = reserve(uint64_t(kDim) * kEmbedCols * 2, gguf::Type::F16, {kEmbedCols, kDim});
        auto* o = reinterpret_cast<__fp16*>(static_cast<char*>(embed_.buf->contents()) + embed_.off);
        for (uint32_t r = 0; r < kDim; ++r)
            for (uint32_t c = 0; c < kEmbedCols; ++c)
                o[uint64_t(r) * kEmbedCols + c] = __fp16(c < kLatent ? w[uint64_t(r) * kLatent + c] : 0.0f);
    }
    embed_b_ = upload_f32(f, "decoder.x_embedder.bias");
    registers_ = upload_f32(f, "decoder.register_tokens");
    if (registers_.dims.empty() || registers_.dims[0] != uint64_t(kRegisters) * kDim)
        throw std::runtime_error("image VAE: not 4 register tokens of 1024");
    ones_ = reserve(kHeadDim * 4, gguf::Type::F32, {kHeadDim});
    std::fill_n(reinterpret_cast<float*>(static_cast<char*>(ones_.buf->contents()) + ones_.off), kHeadDim, 1.0f);
    for (uint32_t i = 0; f.find(std::format("decoder.transformer_blocks.{}.norm1.weight", i)); ++i) {
        const std::string p = std::format("decoder.transformer_blocks.{}.", i);
        Block B;
        B.norm1 = upload_f32(f, p + "norm1.weight");
        B.norm2 = upload_f32(f, p + "norm2.weight");
        B.qkv = upload_f16(f, p + "attn.to_qkv.weight");
        B.qkv_b = upload_f32(f, p + "attn.to_qkv.bias");
        B.o = upload_f16(f, p + "attn.to_out.weight");
        B.o_b = upload_f32(f, p + "attn.to_out.bias");
        B.w1 = upload_f16(f, p + "ff.w1.weight");
        B.w1_b = upload_f32(f, p + "ff.w1.bias");
        B.w2 = upload_f16(f, p + "ff.w2.weight");
        B.w2_b = upload_f32(f, p + "ff.w2.bias");
        B.scale1 = upload_f32(f, p + "scale1");
        B.scale2 = upload_f32(f, p + "scale2");
        if (B.qkv.dims[0] != kDim || B.qkv.dims[1] != 3 * kDim || B.w1.dims[1] != 2 * kFfn || B.w2.dims[0] != kFfn ||
            B.o.dims[0] != kDim || B.o.dims[1] != kDim)
            throw std::runtime_error(p + ": unexpected shapes");
        blocks_.push_back(std::move(B));
    }
    if (blocks_.empty())
        throw std::runtime_error("image VAE: no decoder blocks");
    out_norm_w_ = upload_f32(f, "decoder.norm_out.weight");
    out_norm_b_ = upload_f32(f, "decoder.norm_out.bias");
    proj_out_ = upload_f16(f, "decoder.proj_out.weight");
    proj_out_b_ = upload_f32(f, "decoder.proj_out.bias");
    if (proj_out_.dims[0] != kDim || proj_out_.dims[1] != kOut)
        throw std::runtime_error("image VAE: the patch projection is not 1024 -> 768");
    log::info("image VAE decoder: {} blocks, {:.2f} GB", blocks_.size(), double(dense_bytes_) / 1e9);
}

std::vector<H3ImageVae::Tile> H3ImageVae::tiles(uint32_t length)
{
    // The fewest 256-pixel tiles that cover `length` with overlaps of at least 64, the slack spread over the overlaps
    // in whole latents.
    if (kTile >= length)
        return {{0, length, 0}};
    uint32_t n = (length + kTile - 1) / kTile;
    while (int64_t(kTile) * n - int64_t(kTileOverlap) * (n - 1) - int64_t(length) < 0)
        ++n;
    std::vector<uint32_t> overlaps(n - 1, kTileOverlap);
    const uint32_t remaining = kTile * n - kTileOverlap * (n - 1) - length;
    for (uint32_t i = 0; i < remaining / kPatch; ++i)
        overlaps[i % (n - 1)] += kPatch;
    std::vector<Tile> t{{0, kTile, 0}};
    for (uint32_t i = 1; i < n; ++i)
        t.push_back({t.back().start + kTile - overlaps[i - 1], kTile, overlaps[i - 1]});
    return t;
}

void H3ImageVae::vit(std::span<const float> z, uint32_t th, uint32_t tw, std::vector<float>& out)
{
    const uint32_t N0 = th * tw, N = N0 + kRegisters + 1, Tg = round_up(N, 256), D = kDim;
    const uint32_t q_pad = round_up(N, kAttnRows), k_pad = round_up(N, 128);
    if (z.size() != uint64_t(N0) * kLatent)
        throw std::runtime_error("vit: latents of the tile's shape");
    MTL::Buffer* in = scratch("v_in", uint64_t(N0) * kEmbedCols * 4);
    MTL::Buffer* x = scratch("v_x", uint64_t(Tg) * D * 4);
    MTL::Buffer* xn = scratch("v_xn", uint64_t(Tg) * D * 4);
    MTL::Buffer* qkv = scratch("v_qkv", uint64_t(Tg) * 3 * D * 4);
    MTL::Buffer* qh = scratch("v_qh", uint64_t(kHeads) * q_pad * kHeadDim * 2);
    MTL::Buffer* kh = scratch("v_kh", uint64_t(kHeads) * k_pad * kHeadDim * 2);
    MTL::Buffer* vh = scratch("v_vh", uint64_t(kHeads) * k_pad * kHeadDim * 2);
    MTL::Buffer* att = scratch("v_att", uint64_t(Tg) * D * 4);
    MTL::Buffer* y = scratch("v_y", uint64_t(Tg) * D * 4);
    MTL::Buffer* gu = scratch("v_gu", uint64_t(Tg) * 2 * kFfn * 4);
    MTL::Buffer* ffm = scratch("v_ffm", uint64_t(Tg) * kFfn * 4);
    MTL::Buffer* o = scratch("v_o", uint64_t(Tg) * kOut * 4);
    MTL::Buffer* cs = scratch("v_cs", uint64_t(N) * (kRot / 2) * 8);
    {
        auto* p = static_cast<float*>(in->contents());
        for (uint32_t t = 0; t < N0; ++t) {
            std::copy_n(z.begin() + uint64_t(t) * kLatent, kLatent, p + uint64_t(t) * kEmbedCols);
            std::fill_n(p + uint64_t(t) * kEmbedCols + kLatent, kEmbedCols - kLatent, 0.0f);
        }
    }
    std::memset(x->contents(), 0, x->length());
    // Rotary angles: each axis' coordinate centred on [-1, 1) times 2 pi times 8 frequencies of base 100 (the height
    // axis' pairs, then the width's); the register and zero tokens at the origin.
    {
        auto* c = static_cast<float*>(cs->contents());
        float inv[8];
        for (uint32_t j = 0; j < 8; ++j)
            inv[j] = 1.0f / std::pow(kRopeTheta, float(j) / 8.0f);
        const uint32_t sizes[2] = {th, tw};
        for (uint32_t t = 0; t < N; ++t) {
            float coord[2] = {0, 0};
            if (t < N0) {
                const uint32_t idx[2] = {t / tw, t % tw};
                for (uint32_t a = 0; a < 2; ++a)
                    coord[a] = 2.0f * ((float(idx[a]) + 0.5f) / float(sizes[a])) - 1.0f;
            }
            for (uint32_t a = 0; a < 2; ++a)
                for (uint32_t j = 0; j < 8; ++j) {
                    const float ang = 2.0f * std::numbers::pi_v<float> * coord[a] * inv[j];
                    const uint64_t at = (uint64_t(t) * (kRot / 2) + a * 8 + j) * 2;
                    c[at] = std::cos(ang);
                    c[at + 1] = std::sin(ang);
                }
        }
    }
    MTL::CommandBuffer* cb = ctx_.queue()->commandBuffer();
    cur_cb_ = cb;
    MTL::ComputeCommandEncoder* enc = encoder(cb, 0, -1);
    proj(enc, embed_, in, 0, x, 0, N0);
    bias(enc, x, embed_b_, D, N0);
    {
        const uint32_t n = kRegisters * D;
        dispatch_at(enc, "copy_f32", {registers_, Arg{x, uint64_t(N0) * D * 4}}, &n, sizeof(n), MTL::Size(n, 1, 1),
                    MTL::Size(256, 1, 1));
    }
    for (uint32_t b = 0; b < blocks_.size(); ++b) {
        const Block& B = blocks_[b];
        cur_layer_ = b;
        norm(enc, x, &B.norm1, xn, N, D, D, kEps);
        proj(enc, B.qkv, xn, 0, qkv, 0, Tg);
        bias(enc, qkv, B.qkv_b, 3 * D, N);
        // Each head's q, k and v side by side in a row; q and k RMS-normed (no scale) with their first 32 values
        // rotated.
        heads_partial(enc, qkv, 3 * D, 0, 3 * kHeadDim, kHeadDim, kRot, kHeads, N, q_pad, cs, &ones_, kEps, qh);
        heads_partial(enc, qkv, 3 * D, kHeadDim, 3 * kHeadDim, kHeadDim, kRot, kHeads, N, k_pad, cs, &ones_, kEps, kh);
        heads_partial(enc, qkv, 3 * D, 2 * kHeadDim, 3 * kHeadDim, kHeadDim, 0, kHeads, N, k_pad, cs, nullptr, 0.0f,
                      vh);
        if (k_pad > N) {
            const uint32_t c[4] = {kHeadDim, k_pad, N, 0};
            for (MTL::Buffer* hb : {kh, vh})
                dispatch(enc, "heads_clear", {hb}, c, sizeof(c), MTL::Size(kHeadDim, k_pad - N, kHeads),
                         MTL::Size(64, 1, 1));
        }
        attention(enc, qh, kh, vh, att, N, q_pad, N, k_pad, kHeads, kHeads, 0, 0, kHeadDim);
        proj(enc, B.o, att, 0, y, 0, Tg);
        bias(enc, y, B.o_b, D, N);
        dispatch_at(enc, "scale_add", {x, y, B.scale1}, &D, sizeof(D), MTL::Size(D, N, 1), MTL::Size(256, 1, 1));
        norm(enc, x, &B.norm2, xn, N, D, D, kEps);
        proj(enc, B.w1, xn, 0, gu, 0, Tg);
        bias(enc, gu, B.w1_b, 2 * kFfn, N);
        uint32_t nff = kFfn;
        dispatch(enc, "swiglu_rows", {gu, ffm}, &nff, sizeof(nff), MTL::Size(nff, Tg, 1), MTL::Size(256, 1, 1));
        proj(enc, B.w2, ffm, 0, y, 0, Tg);
        bias(enc, y, B.w2_b, D, N);
        dispatch_at(enc, "scale_add", {x, y, B.scale2}, &D, sizeof(D), MTL::Size(D, N, 1), MTL::Size(256, 1, 1));
        probe(enc, "vit_block", int(b), x, uint64_t(N) * D);
    }
    {
        const LnArgs a{D, kEps};
        enc->setComputePipelineState(pso("layer_norm"));
        enc->setBuffer(x, 0, 0);
        enc->setBuffer(out_norm_w_.buf, out_norm_w_.off, 1);
        enc->setBuffer(out_norm_b_.buf, out_norm_b_.off, 2);
        enc->setBuffer(xn, 0, 3);
        enc->setBytes(&a, sizeof(a), 4);
        enc->dispatchThreadgroups(MTL::Size(N0, 1, 1), MTL::Size(256, 1, 1));
        after(enc, "layer_norm");
    }
    proj(enc, proj_out_, xn, 0, o, 0, Tg);
    bias(enc, o, proj_out_b_, kOut, N0);
    enc->endEncoding();
    finish(cb);
    // Each token's [3][16][16] values to pixels [16 th][16 tw][3].
    const uint32_t H = kPatch * th, W = kPatch * tw;
    out.resize(uint64_t(H) * W * 3);
    const auto* src = static_cast<const float*>(o->contents());
    for (uint32_t t = 0; t < N0; ++t) {
        const uint32_t ty = t / tw, tx = t % tw;
        const float* v = src + uint64_t(t) * kOut;
        for (uint32_t c = 0; c < 3; ++c)
            for (uint32_t py = 0; py < kPatch; ++py) {
                float* dst = &out[((uint64_t(ty) * kPatch + py) * W + tx * kPatch) * 3 + c];
                const float* s = v + (c * kPatch + py) * kPatch;
                for (uint32_t px = 0; px < kPatch; ++px)
                    dst[uint64_t(px) * 3] = s[px];
            }
    }
}

std::vector<float> H3ImageVae::decode(std::span<const float> latents, uint32_t h, uint32_t w)
{
    if (!h || !w || latents.size() != uint64_t(h) * w * kLatent)
        throw std::runtime_error("image VAE: latents [h][w][24]");
    const uint32_t H = h * kPatch, W = w * kPatch;
    // Denormalized and through the post-quant 1x1 convolution.
    std::vector<float> z(latents.size());
    for (uint64_t v = 0; v < latents.size() / kLatent; ++v) {
        float d[kLatent];
        for (uint32_t c = 0; c < kLatent; ++c)
            d[c] = latents[v * kLatent + c] * std_[c] + mean_[c];
        for (uint32_t o = 0; o < kLatent; ++o) {
            float s = pq_b_[o];
            for (uint32_t c = 0; c < kLatent; ++c)
                s += pq_w_[o * kLatent + c] * d[c];
            z[v * kLatent + o] = s;
        }
    }
    const auto rows = tiles(H), cols = tiles(W);
    std::vector<float> canvas(uint64_t(H) * W * 3), tile, lat;
    std::vector<std::vector<float>> row_tiles(cols.size()), prev_row(cols.size());
    // Each tile decoded alone, then stitched: a tile blended with the one above over their overlap and with the one
    // to its left (that one's own pixels), then cropped by the overlaps it shares with the next.
    uint32_t y0 = 0;
    for (size_t i = 0; i < rows.size(); ++i) {
        const uint32_t th = rows[i].size / kPatch, ly = rows[i].start / kPatch;
        const uint32_t keep_h = i + 1 < rows.size() ? rows[i].size - rows[i + 1].overlap : rows[i].size;
        for (size_t j = 0; j < cols.size(); ++j) {
            const uint32_t tw = cols[j].size / kPatch, lx = cols[j].start / kPatch;
            lat.resize(uint64_t(th) * tw * kLatent);
            for (uint32_t yy = 0; yy < th; ++yy)
                std::copy_n(&z[((uint64_t(ly + yy)) * w + lx) * kLatent], uint64_t(tw) * kLatent,
                            &lat[uint64_t(yy) * tw * kLatent]);
            vit(lat, th, tw, row_tiles[j]);
        }
        uint32_t x0 = 0;
        for (size_t j = 0; j < cols.size(); ++j) {
            const uint32_t tH = rows[i].size, tW = cols[j].size;
            tile = row_tiles[j];
            auto at = [&](std::vector<float>& t, uint32_t yy, uint32_t xx) -> float* { return &t[(uint64_t(yy) * tW + xx) * 3]; };
            if (i > 0) { // the tile above: its bottom rows fade into this one's top rows
                const uint32_t ov = rows[i].overlap;
                for (uint32_t yy = 0; yy < ov; ++yy) {
                    const float wb = float(yy) / float(ov), wa = 1.0f - wb;
                    const float* a = at(prev_row[j], tH - ov + yy, 0);
                    float* b = at(tile, yy, 0);
                    for (uint32_t k = 0; k < tW * 3; ++k)
                        b[k] = a[k] * wa + b[k] * wb;
                }
            }
            if (j > 0) { // the tile to the left, as decoded
                const uint32_t ov = cols[j].overlap;
                for (uint32_t yy = 0; yy < tH; ++yy)
                    for (uint32_t xx = 0; xx < ov; ++xx) {
                        const float wb = float(xx) / float(ov), wa = 1.0f - wb;
                        const float* a = at(row_tiles[j - 1], yy, tW - ov + xx);
                        float* b = at(tile, yy, xx);
                        for (uint32_t k = 0; k < 3; ++k)
                            b[k] = a[k] * wa + b[k] * wb;
                    }
            }
            const uint32_t keep_w = j + 1 < cols.size() ? tW - cols[j + 1].overlap : tW;
            for (uint32_t yy = 0; yy < keep_h; ++yy)
                std::copy_n(at(tile, yy, 0), uint64_t(keep_w) * 3, &canvas[((uint64_t(y0) + yy) * W + x0) * 3]);
            x0 += keep_w;
        }
        y0 += keep_h;
        std::swap(prev_row, row_tiles);
    }
    // ImageNet-normalized to [0, 1].
    for (uint64_t i = 0; i < canvas.size(); ++i)
        canvas[i] = std::clamp(canvas[i] * kPixelStd[i % 3] + kPixelMean[i % 3], 0.0f, 1.0f);
    return canvas;
}

void H3ImageVae::load_encoder()
{
    if (enc_)
        return;
    const auto f = open_files(weights_);
    static constexpr uint32_t kMult[] = {1, 2, 2, 4, 4, 8};
    auto conv3 = [&](const std::string& name, uint32_t stride = 1) {
        const std::string n[1] = {name};
        Conv c = load_conv(f, n, 3);
        c.mode = Pad::reflect;
        c.stride = stride;
        c.pad = stride == 2 ? 0 : 1; // the downsample pads below and right only
        return c;
    };
    auto conv1 = [&](const std::string& name) {
        const std::string n[1] = {name};
        return load_conv(f, n, 1);
    };
    auto e = std::make_unique<Encoder>();
    e->in = conv3("encoder.conv_in");
    uint32_t cin = kEncCh;
    for (uint32_t l = 0; l < 6; ++l) {
        EncLevel L;
        const uint32_t cout = kEncCh * kMult[l];
        for (uint32_t b = 0; b < 2; ++b) {
            const std::string p = std::format("encoder.down.{}.block.{}.", l, b);
            EncRes r;
            r.g1 = upload_f32(f, p + "norm1.weight");
            r.b1 = upload_f32(f, p + "norm1.bias");
            r.c1 = conv3(p + "conv1");
            r.g2 = upload_f32(f, p + "norm2.weight");
            r.b2 = upload_f32(f, p + "norm2.bias");
            r.c2 = conv3(p + "conv2");
            r.shortcut = static_cast<bool>(f.find(p + "nin_shortcut.weight"));
            if (r.shortcut)
                r.sc = conv1(p + "nin_shortcut");
            if (r.c1.cin != cin || r.c2.cout != cout)
                throw std::runtime_error(p + ": unexpected widths");
            cin = cout;
            L.res.push_back(std::move(r));
        }
        if (l < 4) {
            L.down = true;
            L.down_conv = conv3(std::format("encoder.down.{}.downsample.conv", l), 2);
        }
        e->levels.push_back(std::move(L));
    }
    e->out_g = upload_f32(f, "encoder.norm_out.weight");
    e->out_b = upload_f32(f, "encoder.norm_out.bias");
    e->out = conv3("encoder.conv_out");
    e->quant = conv1("quant_conv");
    if (e->out.cout != 2 * kLatent || e->quant.cout != 2 * kLatent)
        throw std::runtime_error("image VAE: the encoder does not end in 48 moments");
    enc_ = std::move(e);
    log::info("image VAE encoder loaded ({:.2f} GB resident in all)", double(dense_bytes_) / 1e9);
}

void H3ImageVae::encode_tile(std::span<const float> pixels, uint32_t th, uint32_t tw, std::vector<float>& out)
{
    const uint32_t rows = th * tw;
    MTL::Buffer* x = scratch("e_x", uint64_t(rows) * 3 * 4);
    std::memcpy(x->contents(), pixels.data(), pixels.size() * 4);
    // Channels-last activations; the widest level is the first (128 channels at full resolution).
    const uint64_t cap = uint64_t(rows) * kEncCh * 4;
    MTL::Buffer* h = scratch("e_h", cap);
    MTL::Buffer* t1 = scratch("e_t1", cap);
    MTL::Buffer* t2 = scratch("e_t2", cap);
    MTL::CommandBuffer* cb = ctx_.queue()->commandBuffer();
    cur_cb_ = cb;
    MTL::ComputeCommandEncoder* enc = encoder(cb, 0, -1);
    conv(enc, enc_->in, x, th, tw, h);
    uint32_t hh = th, ww = tw;
    for (size_t l = 0; l < enc_->levels.size(); ++l) {
        const EncLevel& L = enc_->levels[l];
        for (const EncRes& r : L.res) {
            const uint32_t n = hh * ww;
            group_norm(enc, h, r.g1, r.b1, t1, n, r.c1.cin, true);
            conv(enc, r.c1, t1, hh, ww, t2);
            group_norm(enc, t2, r.g2, r.b2, t2, n, r.c2.cin, true);
            conv(enc, r.c2, t2, hh, ww, t1);
            if (r.shortcut) {
                conv(enc, r.sc, h, hh, ww, t2);
                add(enc, t2, t1, h, n * r.c2.cout);
            } else {
                add(enc, h, t1, h, n * r.c2.cout);
            }
        }
        if (L.down) {
            conv(enc, L.down_conv, h, hh, ww, t1);
            std::swap(h, t1);
            hh /= 2;
            ww /= 2;
        }
        probe(enc, "enc_level", int(l), h, uint64_t(hh) * ww * (L.down ? L.down_conv.cout : L.res.back().c2.cout));
    }
    const uint32_t n = hh * ww;
    group_norm(enc, h, enc_->out_g, enc_->out_b, t1, n, enc_->out.cin, true);
    conv(enc, enc_->out, t1, hh, ww, t2);
    conv(enc, enc_->quant, t2, hh, ww, t1);
    enc->endEncoding();
    finish(cb);
    out.resize(uint64_t(n) * 2 * kLatent);
    std::memcpy(out.data(), t1->contents(), out.size() * 4);
}

void H3ImageVae::moments(std::span<const uint8_t> rgb, uint32_t H, uint32_t W, std::vector<float>& out)
{
    if (!H || !W || H % kPatch || W % kPatch || rgb.size() != uint64_t(H) * W * 3)
        throw std::runtime_error("image VAE: an image [H][W][3] with H and W multiples of 16");
    load_encoder();
    const auto rows = tiles(H), cols = tiles(W);
    const uint32_t lh = H / kPatch, lw = W / kPatch, C = 2 * kLatent;
    out.assign(uint64_t(lh) * lw * C, 0.0f);
    std::vector<float> pix, tile;
    std::vector<std::vector<float>> row_tiles(cols.size()), prev_row(cols.size());
    uint32_t y0 = 0;
    for (size_t i = 0; i < rows.size(); ++i) {
        const uint32_t tH = rows[i].size, lth = tH / kPatch;
        for (size_t j = 0; j < cols.size(); ++j) {
            const uint32_t tW = cols[j].size;
            pix.resize(uint64_t(tH) * tW * 3);
            for (uint32_t yy = 0; yy < tH; ++yy)
                for (uint32_t xx = 0; xx < tW; ++xx)
                    for (uint32_t c = 0; c < 3; ++c) {
                        const uint8_t v = rgb[((uint64_t(rows[i].start + yy) * W) + cols[j].start + xx) * 3 + c];
                        pix[(uint64_t(yy) * tW + xx) * 3 + c] = (float(v) / 255.0f - kPixelMean[c]) / kPixelStd[c];
                    }
            encode_tile(pix, tH, tW, row_tiles[j]);
        }
        // Stitched as the reference stitches: each tile blended over its overlap with the raw tile above and the raw
        // one to its left, then cropped by the overlaps it shares with the next tiles.
        uint32_t x0 = 0;
        for (size_t j = 0; j < cols.size(); ++j) {
            const uint32_t ltw = cols[j].size / kPatch;
            tile = row_tiles[j];
            auto at = [&](std::vector<float>& t, uint32_t yy, uint32_t xx) -> float* { return &t[(uint64_t(yy) * ltw + xx) * C]; };
            if (i > 0) {
                const uint32_t ov = rows[i].overlap / kPatch;
                for (uint32_t yy = 0; yy < ov; ++yy) {
                    const float wb = float(yy) / float(ov), wa = 1.0f - wb;
                    const float* a = at(prev_row[j], lth - ov + yy, 0);
                    float* b = at(tile, yy, 0);
                    for (uint32_t k = 0; k < ltw * C; ++k)
                        b[k] = a[k] * wa + b[k] * wb;
                }
            }
            if (j > 0) {
                const uint32_t ov = cols[j].overlap / kPatch;
                for (uint32_t yy = 0; yy < lth; ++yy)
                    for (uint32_t xx = 0; xx < ov; ++xx) {
                        const float wb = float(xx) / float(ov), wa = 1.0f - wb;
                        const float* a = at(row_tiles[j - 1], yy, ltw - ov + xx);
                        float* b = at(tile, yy, xx);
                        for (uint32_t k = 0; k < C; ++k)
                            b[k] = a[k] * wa + b[k] * wb;
                    }
            }
            const uint32_t keep_h = i + 1 < rows.size() ? lth - rows[i + 1].overlap / kPatch : lth;
            const uint32_t keep_w = j + 1 < cols.size() ? ltw - cols[j + 1].overlap / kPatch : ltw;
            for (uint32_t yy = 0; yy < keep_h; ++yy)
                std::copy_n(at(tile, yy, 0), uint64_t(keep_w) * C, &out[((uint64_t(y0) + yy) * lw + x0) * C]);
            x0 += keep_w;
        }
        y0 += i + 1 < rows.size() ? lth - rows[i + 1].overlap / kPatch : lth;
        prev_row.swap(row_tiles);
    }
}

void H3ImageVae::encode_moments(std::span<const uint8_t> rgb, uint32_t H, uint32_t W, std::vector<float>& mean,
                                std::vector<float>& sd)
{
    std::vector<float> m;
    moments(rgb, H, W, m);
    const uint64_t n = m.size() / (2 * kLatent);
    mean.resize(n * kLatent);
    sd.resize(n * kLatent);
    for (uint64_t v = 0; v < n; ++v)
        for (uint32_t c = 0; c < kLatent; ++c) {
            mean[v * kLatent + c] = m[v * 2 * kLatent + c];
            sd[v * kLatent + c] = std::exp(0.5f * std::clamp(m[v * 2 * kLatent + kLatent + c], -30.0f, 20.0f));
        }
}

std::vector<float> H3ImageVae::encode(std::span<const uint8_t> rgb, uint32_t H, uint32_t W, std::span<const float> noise)
{
    std::vector<float> mean, sd;
    encode_moments(rgb, H, W, mean, sd);
    if (noise.size() != mean.size())
        throw std::runtime_error("image VAE: the noise is not [H / 16][W / 16][24]");
    std::vector<float> z(mean.size());
    for (uint64_t i = 0; i < z.size(); ++i) {
        const float v = float(__fp16(mean[i] + sd[i] * noise[i])); // the sample kept to fp16
        z[i] = (v - mean_[i % kLatent]) / std_[i % kLatent];
    }
    return z;
}

} // namespace mako::image
