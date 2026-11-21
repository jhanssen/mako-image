#include "image/qwen_image.hpp"

#include "log/log.hpp"
#include "kernels/args.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <format>
#include <stdexcept>

namespace mako::image {

namespace {

using kernels::HeadsArgs;
struct ModArgs {
    uint32_t n;
    float eps;
    uint32_t split, mod_stride, mod_off;
};
using kernels::HeadRowsArgs;

// Pairs i of the rotary angle table for (frame, height, width) positions: 8 pairs on the frame axis, then 28 each on
// height and width (axes 16 / 56 / 56 of the 128 dims), theta 10000.
void rope3(float* cs, int64_t f, int64_t hpos, int64_t wpos)
{
    constexpr int dims[3] = {16, 56, 56};
    const int64_t pos[3] = {f, hpos, wpos};
    uint32_t i = 0;
    for (int a = 0; a < 3; ++a)
        for (int j = 0; j < dims[a] / 2; ++j, ++i) {
            const float inv = 1.0f / std::pow(10000.0f, float(2 * j) / float(dims[a]));
            const float ang = float(pos[a]) * inv;
            cs[2 * i] = std::cos(ang);
            cs[2 * i + 1] = std::sin(ang);
        }
}

safetensors::Files open_files(const std::string& path)
{
    auto f = safetensors::Files::open(path);
    if (!f)
        throw std::runtime_error(f.error());
    return std::move(*f);
}

} // namespace

QwenImage::QwenImage(const std::string& dir, mtl::Context& ctx, const Options& opts) : Engine(dir, ctx, opts)
{
    if (text_width() != kDim)
        throw std::runtime_error("text encoder: not 4096 wide");
    load_transformer(open_files(opts.transformer.empty() ? dir + "/transformer" : opts.transformer));
    drop_ = uint32_t(tok_->encode("<|im_start|>system\n" + std::string(kSystem) + "<|im_end|>\n", true).size());
    vision_dir_ = dir + "/text_encoder"; // the release's tower, whichever language model the text encoder is
}

std::string QwenImage::templated(const std::string& prompt)
{
    return "<|im_start|>system\n" + std::string(kSystem) + "<|im_end|>\n<|im_start|>user\n" +
           (prompt.empty() ? " " : prompt) + "<|im_end|>\n<|im_start|>assistant\n";
}

void QwenImage::load_transformer(const safetensors::Files& f)
{
    img_in_ = upload(f, "img_in.weight", true);
    txt_norm_ = upload(f, "txt_in.text_norm.weight", false);
    // Stored zero-centred: the scale is weight + 1.
    auto* tn = reinterpret_cast<float*>(static_cast<char*>(txt_norm_.buf->contents()) + txt_norm_.off);
    for (uint32_t i = 0; i < kDim; ++i)
        tn[i] += 1.0f;
    txt_in1_ = upload(f, "txt_in.in_layer.weight", true);
    txt_in2_ = upload(f, "txt_in.out_layer.weight", true);
    t_lin1_ = upload(f, "time_text_embed.timestep_embedder.linear_1.weight", true);
    t_lin2_ = upload(f, "time_text_embed.timestep_embedder.linear_2.weight", true);
    mod_ = upload(f, "modulation.1.weight", true);
    out_lin_ = upload(f, "norm_out.linear.weight", true);
    proj_out_ = upload(f, "proj_out.weight", true);
    for (uint32_t i = 0;; ++i) {
        const std::string b = std::format("transformer_blocks.{}.", i);
        if (!f.find(b + "attn.to_q.weight"))
            break;
        Block B;
        const std::string qkv[3] = {b + "attn.to_q.weight", b + "attn.to_k.weight", b + "attn.to_v.weight"};
        B.qkv = upload_rows(f, qkv);
        B.o = upload(f, b + "attn.to_out.0.weight", true);
        B.norm_q = upload(f, b + "attn.norm_q.weight", false);
        B.norm_k = upload(f, b + "attn.norm_k.weight", false);
        const std::string gu[2] = {b + "img_mlp.gate_layer.weight", b + "img_mlp.proj.weight"};
        B.gate_up = upload_rows(f, gu);
        B.down = upload(f, b + "img_mlp.out.weight", true);
        blocks_.push_back(std::move(B));
    }
    if (blocks_.empty())
        throw std::runtime_error("transformer: no blocks");
}

std::pair<uint32_t, uint32_t> QwenImage::reference_size(uint32_t w, uint32_t h)
{
    const double ratio = double(w) / double(h), width = std::sqrt(double(kReferenceArea) * ratio);
    auto side = [](double v) { return std::max<uint32_t>(32, uint32_t(std::lround(v / 32.0)) * 32); };
    return {side(width), side(width / ratio)};
}

models::QwenVision& QwenImage::vision()
{
    if (!vision_)
        vision_ = std::make_unique<models::QwenVision>(vision_dir_, ctx_, tel_);
    return *vision_;
}

void QwenImage::set_references(const Request& r)
{
    refs_.clear();
    slots_.clear();
    for (size_t i = 0; i < r.references.size(); ++i) {
        const Request::Reference& src = r.references[i];
        if (!src.width || !src.height || src.rgba.size() != uint64_t(src.width) * src.height * 4)
            throw std::runtime_error(std::format("reference {}: RGBA [height][width][4]", i + 1));
        const auto [w, h] = reference_size(src.width, src.height);
        std::vector<float> sized;
        std::span<const float> rgba = src.rgba;
        if (w != src.width || h != src.height) {
            sized = resize_image(src.rgba, src.width, src.height, w, h);
            rgba = sized;
        }
        // The vision tower sees the picture over white, the VAE all four channels.
        std::vector<uint8_t> rgb(uint64_t(w) * h * 3);
        for (uint64_t p = 0; p < uint64_t(w) * h; ++p) {
            const float a = std::clamp(rgba[p * 4 + 3] * 0.5f + 0.5f, 0.0f, 1.0f);
            for (uint32_t c = 0; c < 3; ++c) {
                const float v = std::clamp(rgba[p * 4 + c] * 0.5f + 0.5f, 0.0f, 1.0f);
                rgb[p * 3 + c] = uint8_t(std::lround((v * a + 1.0f - a) * 255.0f));
            }
        }
        Ref ref;
        ref.lh = h / 16;
        ref.lw = w / 16;
        ref.features = vision().encode(rgb, h, w);
        if (ref.features.rows * 2 != ref.lh || ref.features.cols * 2 != ref.lw)
            throw std::logic_error("a vision token covers 2x2 latents");
        ref.latents = encode(rgba, ref.lh, ref.lw);
        refs_.push_back(std::move(ref));
    }
}

void QwenImage::set_reference_latents(size_t i, std::vector<float> latents)
{
    Ref& r = refs_.at(i);
    if (latents.size() != uint64_t(r.lh) * r.lw * kLatent)
        throw std::runtime_error(std::format("reference {}: latents [{}][{}][64]", i + 1, r.lh, r.lw));
    r.latents = std::move(latents);
}

std::vector<float> QwenImage::prompt_embeddings(const std::string& prompt)
{
    slots_.clear();
    if (refs_.empty()) {
        auto ids = tok_->encode(templated(prompt), true);
        if (ids.size() > drop_ + kMaxPrompt) // cut at the end, as the reference's tokenizer does
            ids.resize(drop_ + kMaxPrompt);
        ids_ = ids;
        return encode_text(ids, drop_);
    }
    // The reference's template for pictures: each as "<imageN>" and a vision block before the prompt. The pad
    // tokens' rows leave the embeddings (the transformer puts the picture's latents there); where each went is kept.
    auto special = [&](const char* name) {
        const auto v = tok_->encode(name, true);
        if (v.size() != 1)
            throw std::runtime_error(std::format("the tokenizer has no {} token", name));
        return v[0];
    };
    auto ids = tok_->encode("<|im_start|>system\n" + std::string(kSystem) + "<|im_end|>\n<|im_start|>user\n", true);
    std::vector<models::QwenText::Vision> blocks;
    std::vector<std::pair<uint32_t, uint32_t>> pads; // each picture's pad rows: the first, how many
    for (size_t i = 0; i < refs_.size(); ++i) {
        const auto label = tok_->encode(std::format("{}<image{}>", i ? " " : "", i + 1), false);
        ids.insert(ids.end(), label.begin(), label.end());
        ids.push_back(special("<|vision_start|>"));
        const models::QwenVision::Features& f = refs_[i].features;
        const uint32_t n = f.rows * f.cols;
        models::QwenText::Vision b;
        b.first = uint32_t(ids.size());
        b.rows = f.rows;
        b.cols = f.cols;
        b.embeds = f.embeds;
        for (const auto& d : f.deepstack)
            b.deepstack.emplace_back(d);
        blocks.push_back(std::move(b));
        pads.emplace_back(uint32_t(ids.size()), n);
        ids.insert(ids.end(), n, special("<|image_pad|>"));
        ids.push_back(special("<|vision_end|>"));
    }
    const auto tail = tok_->encode((prompt.empty() ? " " : prompt) + "<|im_end|>\n<|im_start|>assistant\n", true);
    ids.insert(ids.end(), tail.begin(), tail.end());
    ids_ = ids;
    const auto all = encode_text(ids, drop_, {}, false, blocks); // the rows from drop_ on
    std::vector<float> out;
    out.reserve(all.size());
    uint32_t at = drop_;
    for (const auto& [first, n] : pads) {
        out.insert(out.end(), all.begin() + uint64_t(at - drop_) * kDim, all.begin() + uint64_t(first - drop_) * kDim);
        slots_.push_back(uint32_t(out.size() / kDim));
        at = first + n;
    }
    out.insert(out.end(), all.begin() + uint64_t(at - drop_) * kDim, all.end());
    return out;
}

void QwenImage::begin(std::span<const float> prompt, uint32_t h, uint32_t w, uint32_t slot)
{
    if (slot > 1)
        throw std::runtime_error("slot must be 0 or 1");
    Text& g = texts_[slot];
    if (prompt.empty() || prompt.size() % kDim)
        throw std::runtime_error("prompt embeddings must be rows of 4096");
    if (slots_.size() != refs_.size())
        throw std::runtime_error("the prompt embeddings do not place the references (prompt_embeddings() does)");
    g.S = uint32_t(prompt.size() / kDim);
    gh_ = h;
    gw_ = w;
    g.steps = 0;
    g.prompt.assign(prompt.begin(), prompt.end());
    // The prefix: the prompt's rows with each reference's latent rows at its slot.
    g.segments.clear();
    uint32_t row = 0, at = 0;
    for (size_t i = 0; i < refs_.size(); ++i) {
        const uint32_t to = std::min(slots_[i], g.S);
        if (to > at) {
            g.segments.push_back({false, row, to - at, at});
            row += to - at;
            at = to;
        }
        const uint32_t n = refs_[i].lh * refs_[i].lw;
        g.segments.push_back({true, row, n, uint32_t(i)});
        row += n;
    }
    if (g.S > at) {
        g.segments.push_back({false, row, g.S - at, at});
        row += g.S - at;
    }
    g.P = row;
    g.k.clear();
    g.v.clear();
    for (size_t b = 0; b < blocks_.size(); ++b) {
        const uint64_t bytes = uint64_t(kHeads) * g.P * kHeadDim * 2;
        g.k.emplace_back(ctx_.device()->newBuffer(bytes, MTL::ResourceStorageModeShared));
        g.v.emplace_back(ctx_.device()->newBuffer(bytes, MTL::ResourceStorageModeShared));
    }
    // Rotary positions: text rows advance a shared position on all three axes; an image block (a reference, and the
    // generated image last) takes the frame position next with a height / width grid centred on zero, and advances
    // the position by its larger side.
    const uint32_t N = h * w;
    MTL::Buffer* cs = scratch(std::format("d_cs{}", slot), uint64_t(g.P + N) * kHeadDim / 2 * 8);
    auto* c = static_cast<float*>(cs->contents());
    int64_t pos = 0;
    auto grid = [&](uint32_t row0, uint32_t gh, uint32_t gw) {
        for (uint32_t y = 0; y < gh; ++y)
            for (uint32_t x = 0; x < gw; ++x)
                rope3(c + uint64_t(row0 + y * gw + x) * kHeadDim, pos, int64_t(y) - int64_t(gh - gh / 2),
                      int64_t(x) - int64_t(gw - gw / 2));
        pos += std::max(gh, gw);
    };
    for (const Segment& sg : g.segments) {
        if (sg.image) {
            grid(sg.row0, refs_[sg.src].lh, refs_[sg.src].lw);
        } else {
            for (uint32_t t = 0; t < sg.n; ++t, ++pos)
                rope3(c + uint64_t(sg.row0 + t) * kHeadDim, pos, pos, pos);
        }
    }
    grid(g.P, h, w);
}

std::vector<float> QwenImage::step(std::span<const float> latents, float sigma, uint32_t slot)
{
    if (slot > 1)
        throw std::runtime_error("slot must be 0 or 1");
    Text& g = texts_[slot];
    const uint32_t N = gh_ * gw_, D = kDim;
    if (latents.size() != uint64_t(N) * kLatent)
        throw std::runtime_error("latents must be [h * w][64]");
    const bool first = g.steps++ == 0;
    // The first step runs the prefix rows too (the prompt's and the references'); later ones only the image's,
    // against the prefix's cached keys and values.
    const uint32_t S = g.S, P = g.P, T = first ? P + N : N, row0 = first ? 0 : P, n_k = P + N;
    const uint32_t q_pad = round_up(T, kAttnRows), k_pad = round_up(n_k, 128), split = first ? P : 0;
    // The GEMMs run a multiple of 256 rows (the rest ignored): fewer rows, as the first step's text and image make,
    // would take 64-row tiles that read the weights four times as often (that step took twice as long).
    const uint32_t Tg = T > 256 ? round_up(T, 256) : T;
    MTL::Buffer* x = scratch("d_x", uint64_t(Tg) * D * 4);
    MTL::Buffer* xn = scratch("d_xn", uint64_t(Tg) * D * 4);
    MTL::Buffer* qkv = scratch("d_qkv", uint64_t(Tg) * 3 * D * 4);
    const uint64_t nff = blocks_.front().down.dims[0];
    MTL::Buffer* gu = scratch("d_gu", uint64_t(Tg) * 2 * nff * 4);
    MTL::Buffer* ffm = scratch("d_ffm", uint64_t(Tg) * nff * 4);
    MTL::Buffer* qh = scratch("d_qh", uint64_t(kHeads) * q_pad * kHeadDim * 2);
    MTL::Buffer* kh = scratch("d_kh", uint64_t(kHeads) * k_pad * kHeadDim * 2);
    MTL::Buffer* vh = scratch("d_vh", uint64_t(kHeads) * k_pad * kHeadDim * 2);
    MTL::Buffer* att = scratch("d_att", uint64_t(Tg) * D * 4);
    MTL::Buffer* y = scratch("d_y", uint64_t(Tg) * D * 4);
    MTL::Buffer* lat = scratch("d_lat", uint64_t(N) * kLatent * 4);
    MTL::Buffer* out = scratch("d_out", uint64_t(Tg) * kLatent * 4);
    MTL::Buffer* tp = scratch("d_tproj", 2 * kTimeDim * 4);
    MTL::Buffer* temb = scratch("d_temb", 2 * D * 4);
    MTL::Buffer* temb_s = scratch("d_temb_s", 2 * D * 4);
    MTL::Buffer* mod = scratch("d_mod", 2 * 4 * D * 4);
    MTL::Buffer* omod = scratch("d_omod", 2 * D * 4);
    MTL::Buffer* cs = scratch(std::format("d_cs{}", slot), uint64_t(P + N) * kHeadDim / 2 * 8);
    std::memcpy(lat->contents(), latents.data(), latents.size() * 4);
    // Sinusoidal timestep embeddings (cos half first) of 1000 * sigma and of 0, the prefix's.
    auto* tpf = static_cast<float*>(tp->contents());
    for (int r = 0; r < 2; ++r)
        for (uint32_t i = 0; i < kTimeDim / 2; ++i) {
            const float freq = std::exp(-std::log(10000.0f) * float(i) / float(kTimeDim / 2));
            const float arg = (r == 0 ? 1000.0f * sigma : 0.0f) * freq;
            tpf[r * kTimeDim + i] = std::cos(arg);
            tpf[r * kTimeDim + kTimeDim / 2 + i] = std::sin(arg);
        }
    std::vector<MTL::Buffer*> ref_lat(refs_.size(), nullptr);
    MTL::Buffer* limits = nullptr;
    if (first) {
        MTL::Buffer* pr = scratch("d_prompt", uint64_t(S) * D * 4);
        std::memcpy(pr->contents(), g.prompt.data(), g.prompt.size() * 4);
        for (size_t i = 0; i < refs_.size(); ++i) {
            ref_lat[i] = scratch(std::format("d_ref{}", i), refs_[i].latents.size() * 4);
            std::memcpy(ref_lat[i]->contents(), refs_[i].latents.data(), refs_[i].latents.size() * 4);
        }
        // What each query row sees: a text row the rows before it, a reference's rows those before their block's
        // end, the image's rows everything (without references the kernel's own rule says the same).
        if (!refs_.empty()) {
            limits = scratch("d_limits", uint64_t(q_pad) * 4);
            auto* l = static_cast<uint32_t*>(limits->contents());
            for (const Segment& sg : g.segments)
                for (uint32_t i = 0; i < sg.n; ++i)
                    l[sg.row0 + i] = sg.image ? sg.row0 + sg.n : sg.row0 + i + 1;
            for (uint32_t i = P; i < q_pad; ++i)
                l[i] = n_k;
        }
    }

    MTL::CommandBuffer* cb = ctx_.queue()->commandBuffer();
    cur_cb_ = cb;
    MTL::ComputeCommandEncoder* enc = encoder(cb, 0, -1);
    proj(enc, t_lin1_, tp, 0, temb, 0, 2);
    uint32_t n2 = 2 * D;
    dispatch(enc, "silu", {temb}, &n2, sizeof(n2), MTL::Size(n2, 1, 1), MTL::Size(256, 1, 1));
    proj(enc, t_lin2_, temb, 0, temb_s, 0, 2);
    probe(enc, "temb", -1, temb_s, 2 * D);
    // silu(temb) feeds the block modulation and the final norm's.
    dispatch(enc, "silu", {temb_s}, &n2, sizeof(n2), MTL::Size(n2, 1, 1), MTL::Size(256, 1, 1));
    proj(enc, mod_, temb_s, 0, mod, 0, 2);
    probe(enc, "modulation", -1, mod, 2 * 4 * D);
    proj(enc, out_lin_, temb_s, 0, omod, 0, 2);
    if (first) {
        // The prompt rows through the text projection, each segment to its rows; a reference's latents through the
        // image projection to theirs.
        MTL::Buffer* pr = scratch("d_prompt", uint64_t(S) * D * 4);
        norm(enc, pr, &txt_norm_, xn, S, D, D, kEps);
        proj(enc, txt_in1_, xn, 0, y, 0, S);
        uint32_t n = S * D;
        dispatch(enc, "gelu_tanh", {y}, &n, sizeof(n), MTL::Size(n, 1, 1), MTL::Size(256, 1, 1));
        for (const Segment& sg : g.segments) {
            if (sg.image)
                proj(enc, img_in_, ref_lat[sg.src], 0, x, uint64_t(sg.row0) * D * 4, sg.n);
            else
                proj(enc, txt_in2_, y, uint64_t(sg.src) * D * 4, x, uint64_t(sg.row0) * D * 4, sg.n);
        }
        if (refs_.empty())
            probe(enc, "txt_in", -1, x, uint64_t(S) * D);
    }
    proj(enc, img_in_, lat, 0, x, uint64_t(T - N) * D * 4, N);
    probe(enc, "img_in", -1, x, uint64_t(N) * D, uint64_t(T - N) * D * 4);

    const uint32_t mstride = 4 * D;
    for (uint32_t il = 0; il < blocks_.size(); ++il) {
        const Block& B = blocks_[il];
        cur_layer_ = il;
        ModArgs ma{D, kEps, split, mstride, 0};
        dispatch(enc, "ln_modulate", {x, mod, xn}, &ma, sizeof(ma), MTL::Size(T, 1, 1), MTL::Size(256, 1, 1), false);
        proj(enc, B.qkv, xn, 0, qkv, 0, Tg);
        // q and k RMS-normed per head and rotated (the table's rows from this step's first token on), v as it is.
        const uint64_t cs_off = uint64_t(row0) * kHeadDim / 2 * 8;
        heads_norm(enc, qkv, 3 * D, 0, kHeads, T, q_pad, cs, cs_off, 2, B.norm_q, kEps, 0, qh);
        heads_norm(enc, qkv, 3 * D, D, kHeads, T, k_pad, cs, cs_off, 2, B.norm_k, kEps, row0, kh);
        HeadsArgs ha{T, 3 * D, 2 * D, kHeadDim, k_pad, row0, 0};
        enc->setComputePipelineState(pso("heads_half"));
        enc->setBuffer(qkv, 0, 0);
        enc->setBuffer(cs, 0, 1);
        enc->setBuffer(vh, 0, 2);
        enc->setBytes(&ha, sizeof(ha), 3);
        enc->dispatchThreads(MTL::Size(kHeadDim / 2, T, kHeads), MTL::Size(64, 1, 1));
        after(enc, "heads_half");
        HeadRowsArgs hr{kHeadDim, k_pad, P, P};
        if (first) { // keep the prefix's keys and values
            dispatch(enc, "heads_rows", {kh, g.k[il].get()}, &hr, sizeof(hr), MTL::Size(kHeadDim, P, kHeads),
                     MTL::Size(128, 1, 1));
            dispatch(enc, "heads_rows", {vh, g.v[il].get()}, &hr, sizeof(hr), MTL::Size(kHeadDim, P, kHeads),
                     MTL::Size(128, 1, 1));
        } else {
            hr = {kHeadDim, P, k_pad, P};
            dispatch(enc, "heads_rows", {g.k[il].get(), kh}, &hr, sizeof(hr), MTL::Size(kHeadDim, P, kHeads),
                     MTL::Size(128, 1, 1));
            dispatch(enc, "heads_rows", {g.v[il].get(), vh}, &hr, sizeof(hr), MTL::Size(kHeadDim, P, kHeads),
                     MTL::Size(128, 1, 1));
        }
        if (k_pad > n_k) {
            const uint32_t c[4] = {kHeadDim, k_pad, n_k, 0};
            dispatch(enc, "heads_clear", {kh}, c, sizeof(c), MTL::Size(kHeadDim, k_pad - n_k, kHeads),
                     MTL::Size(128, 1, 1));
            dispatch(enc, "heads_clear", {vh}, c, sizeof(c), MTL::Size(kHeadDim, k_pad - n_k, kHeads),
                     MTL::Size(128, 1, 1));
        }
        attention(enc, qh, kh, vh, att, T, q_pad, n_k, k_pad, kHeads, kHeads, row0, split, kHeadDim, limits);
        proj(enc, B.o, att, 0, y, 0, Tg);
        ma.mod_off = D;
        dispatch(enc, "gate_add", {x, y, mod}, &ma, sizeof(ma), MTL::Size(D, T, 1), MTL::Size(256, 1, 1));
        ma.mod_off = 2 * D;
        dispatch(enc, "ln_modulate", {x, mod, xn}, &ma, sizeof(ma), MTL::Size(T, 1, 1), MTL::Size(256, 1, 1), false);
        proj(enc, B.gate_up, xn, 0, gu, 0, Tg);
        uint32_t nff = uint32_t(B.down.dims[0]);
        dispatch(enc, "swiglu_rows", {gu, ffm}, &nff, sizeof(nff), MTL::Size(nff, Tg, 1), MTL::Size(256, 1, 1));
        proj(enc, B.down, ffm, 0, y, 0, Tg);
        ma.mod_off = 3 * D;
        dispatch(enc, "gate_add", {x, y, mod}, &ma, sizeof(ma), MTL::Size(D, T, 1), MTL::Size(256, 1, 1));
        probe(enc, "block", int(il), x, uint64_t(T) * D);
    }
    ModArgs oa{D, kEps, split, D, 0};
    dispatch(enc, "ln_modulate", {x, omod, xn}, &oa, sizeof(oa), MTL::Size(T, 1, 1), MTL::Size(256, 1, 1), false);
    probe(enc, "norm_out", -1, xn, uint64_t(T) * D);
    proj(enc, proj_out_, xn, 0, out, 0, Tg);
    enc->endEncoding();
    finish(cb);
    const auto* o = static_cast<const float*>(out->contents()) + uint64_t(T - N) * kLatent;
    return std::vector<float>(o, o + uint64_t(N) * kLatent);
}

Engine::LoraTarget QwenImage::lora_target(const std::string& m)
{
    const uint32_t D = kDim, nff = uint32_t(blocks_.front().down.dims[0]);
    unsigned b = 0;
    int used = 0;
    if (std::sscanf(m.c_str(), "transformer_blocks.%u.%n", &b, &used) == 1 && used > 0 && b < blocks_.size()) {
        Block& B = blocks_[b];
        const std::string_view rest = std::string_view(m).substr(size_t(used));
        if (rest == "attn.to_q")
            return {&B.qkv, 0, D};
        if (rest == "attn.to_k")
            return {&B.qkv, D, D};
        if (rest == "attn.to_v")
            return {&B.qkv, 2 * D, D};
        if (rest == "attn.to_out.0")
            return {&B.o, 0, D};
        if (rest == "img_mlp.gate_layer")
            return {&B.gate_up, 0, nff};
        if (rest == "img_mlp.proj")
            return {&B.gate_up, nff, nff};
        if (rest == "img_mlp.gate_up") // gate rows first
            return {&B.gate_up, 0, 2 * nff};
        if (rest == "img_mlp.out")
            return {&B.down, 0, D};
        return {};
    }
    Tensor* t = m == "img_in" ? &img_in_
                : m == "txt_in.in_layer" ? &txt_in1_
                : m == "txt_in.out_layer" ? &txt_in2_
                : m == "proj_out" ? &proj_out_
                : m == "norm_out.linear" ? &out_lin_
                : m == "modulation.1" ? &mod_
                : m == "time_text_embed.timestep_embedder.linear_1" ? &t_lin1_
                : m == "time_text_embed.timestep_embedder.linear_2" ? &t_lin2_
                                                                     : nullptr;
    return t ? LoraTarget{t, 0, uint32_t(t->dims[1])} : LoraTarget{};
}

} // namespace mako::image
