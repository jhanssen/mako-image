#include "image/sdxl.hpp"

#include "gguf/read_pool.hpp"
#include "log/log.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <format>
#include <stdexcept>
#include <tuple>

namespace mako::image {

namespace {

namespace fs = std::filesystem;

struct HeadsArgs {
    uint32_t rows, ld, col0, dim, n_pad, row0, rope;
};
struct AttnArgs {
    uint32_t n_q, q_pad, n_k, k_pad, heads, kv_heads, q_pos0, block, width;
    float scale;
};
struct LnArgs {
    uint32_t n;
    float eps;
};
struct UpArgs {
    uint32_t h, w, c, c_out, ft, repeats;
};

constexpr uint32_t kD = 64;        // attention heads' width, text encoders' and UNet's
constexpr uint32_t kQRows = 64;    // query rows a threadgroup of the attention kernel
constexpr uint32_t kCtxWidth = 2048, kPooled = 1280, kAdded = 2816, kTokens = 77;
constexpr const char* kUnet = "model.diffusion_model.";
constexpr const char* kClipL = "conditioner.embedders.0.transformer.text_model.";
constexpr const char* kClipG = "conditioner.embedders.1.model.";
constexpr float kLatentScale = 0.13025f;

uint32_t round_up(uint32_t x, uint32_t m) { return (x + m - 1) / m * m; }

// The empty prompt's chunk: start, end, then padding.
std::vector<int32_t> empty_ids(int32_t pad)
{
    std::vector<int32_t> ids(kTokens, pad);
    ids[0] = ClipTokenizer::kStart;
    ids[1] = ClipTokenizer::kEnd;
    return ids;
}

// A LoRA module's name with "." for "_" (Kohya's names have "_" for both).
std::string lora_key(std::string s)
{
    std::replace(s.begin(), s.end(), '_', '.');
    return s;
}

safetensors::Files open_files(const std::string& path)
{
    auto f = safetensors::Files::open(path);
    if (!f)
        throw std::runtime_error(f.error());
    return std::move(*f);
}

// [cos(v f_i)..., sin(v f_i)...] with f_i = 10000^(-i / (dim / 2)), in float as the reference computes it.
void sinusoid(float v, uint32_t dim, float* out)
{
    const uint32_t half = dim / 2;
    for (uint32_t i = 0; i < half; ++i) {
        const float f = std::exp(-std::log(10000.0f) * float(i) / float(half));
        out[i] = std::cos(v * f);
        out[half + i] = std::sin(v * f);
    }
}

// A token's latents (channel, row, column of its 2x2 patch) and the UNet's channels-last image, [2h][2w][4].
void unpack(std::span<const float> tokens, uint32_t h, uint32_t w, float scale, float* img)
{
    for (uint32_t y = 0; y < 2 * h; ++y)
        for (uint32_t x = 0; x < 2 * w; ++x)
            for (uint32_t c = 0; c < 4; ++c)
                img[(uint64_t(y) * 2 * w + x) * 4 + c] =
                    tokens[(uint64_t(y / 2) * w + x / 2) * 16 + c * 4 + (y % 2) * 2 + x % 2] * scale;
}

} // namespace

std::string Sdxl::checkpoint(const std::string& dir)
{
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(dir, ec)) {
        if (e.path().extension() != ".safetensors")
            continue;
        auto f = safetensors::File::open(e.path().string());
        if (!f)
            continue;
        bool unet = false, g = false;
        for (const auto& t : f->tensors()) {
            unet = unet || t.name == std::string(kUnet) + "input_blocks.0.0.weight";
            g = g || t.name == std::string(kClipG) + "ln_final.weight";
        }
        if (unet && g)
            return e.path().string();
    }
    return "";
}

Sdxl::Sdxl(const std::string& dir, mtl::Context& ctx, const Options& opts)
    : Engine(ctx, opts), files_(open_files(opts.transformer.empty() ? checkpoint(dir) : opts.transformer))
{
    dir_ = dir;
    // F16 matrices have only the GEMM, so the embeddings' single rows take it too.
    gemm_min_tokens_ = 1;
    const std::string ckpt = opts.transformer.empty() ? checkpoint(dir) : opts.transformer;
    const fs::path tdir = fs::path(dir) / "tokenizer";
    if (!fs::exists(tdir / "vocab.json") || !fs::exists(tdir / "merges.txt"))
        throw std::runtime_error(std::format("{}: no tokenizer/vocab.json and merges.txt (hf download "
                                             "stabilityai/stable-diffusion-xl-base-1.0 --include 'tokenizer/*' "
                                             "--local-dir {})",
                                             dir, dir));
    auto t = ClipTokenizer::load((tdir / "vocab.json").string(), (tdir / "merges.txt").string());
    if (!t)
        throw std::runtime_error(t.error());
    clip_tok_ = std::move(*t);
    load_clip(0);
    load_clip(1);
    load_unet();
    load_vae_ldm(ckpt, "first_stage_model.", kLatentScale);
    // The discrete schedule: betas linear in their square root from 0.00085 to 0.012 over 1000 steps.
    double cum = 1.0;
    const double b0 = std::sqrt(0.00085), b1 = std::sqrt(0.012);
    for (uint32_t i = 0; i < 1000; ++i) {
        const double b = b0 + (b1 - b0) * double(i) / 999.0;
        cum *= 1.0 - b * b;
        const double s = std::sqrt((1.0 - cum) / cum);
        sigma_[i] = float(s);
        log_sigma_[i] = float(std::log(s));
    }
    lora_merged();
    log::info("SDXL: {} UNet transformer blocks, text encoders {} and {} wide", n_tblocks_, clip_[0].E, clip_[1].E);
}

std::optional<std::string> Sdxl::default_negative() const
{
    return "bad quality, worst quality, worst detail, sketch, censor";
}

Engine::Tensor Sdxl::matrix(const std::string& name)
{
    const auto r = files_.find(name);
    if (!r || r.tensor->dims.size() != 2)
        throw std::runtime_error("missing matrix " + name);
    if (r.tensor->type != gguf::Type::F16)
        return upload(files_, name, true);
    Tensor t = reserve(r.tensor->bytes, gguf::Type::F16, r.tensor->dims);
    gguf::read_exact(r.file->fd(), static_cast<char*>(t.buf->contents()) + t.off, r.tensor->bytes, r.tensor->offset);
    return t;
}

void Sdxl::load_clip(int which)
{
    Clip& C = clip_[size_t(which)];
    const std::string p = which == 0 ? kClipL : kClipG;
    const std::string layer = which == 0 ? "encoder.layers.{}." : "transformer.resblocks.{}.";
    uint32_t n = 0;
    while (files_.find(p + std::vformat(layer, std::make_format_args(n)) + (which == 0 ? "layer_norm1.weight" : "ln_1.weight")))
        ++n;
    if (n < 2)
        throw std::runtime_error(std::format("text encoder {}: no layers", which));
    C.layers_total = n;
    // ViT-L's last layer never runs (its second to last layer's states are the output) but LoRAs adapt it too.
    for (uint32_t i = 0; i < n; ++i) {
        const std::string l = p + std::vformat(layer, std::make_format_args(i));
        auto lm = [&](const std::string& s) { return matrix(l + s); };
        auto lv = [&](const std::string& s) { return upload_f32(files_, l + s); };
        ClipLayer L;
        if (which == 0) {
            L.ln1g = lv("layer_norm1.weight");
            L.ln1b = lv("layer_norm1.bias");
            const std::string qkv[3] = {l + "self_attn.q_proj.weight", l + "self_attn.k_proj.weight",
                                        l + "self_attn.v_proj.weight"};
            L.qkv = upload_rows(files_, qkv);
            std::vector<float> b;
            for (const char* s : {"self_attn.q_proj.bias", "self_attn.k_proj.bias", "self_attn.v_proj.bias"}) {
                const auto v = floats(files_, l + s);
                b.insert(b.end(), v.begin(), v.end());
            }
            L.qkv_b = reserve(b.size() * 4, gguf::Type::F32, {b.size()});
            std::memcpy(static_cast<char*>(L.qkv_b.buf->contents()) + L.qkv_b.off, b.data(), b.size() * 4);
            L.o = lm("self_attn.out_proj.weight");
            L.o_b = lv("self_attn.out_proj.bias");
            L.ln2g = lv("layer_norm2.weight");
            L.ln2b = lv("layer_norm2.bias");
            L.fc1 = lm("mlp.fc1.weight");
            L.fc1_b = lv("mlp.fc1.bias");
            L.fc2 = lm("mlp.fc2.weight");
            L.fc2_b = lv("mlp.fc2.bias");
        } else {
            L.ln1g = lv("ln_1.weight");
            L.ln1b = lv("ln_1.bias");
            L.qkv = lm("attn.in_proj_weight");
            L.qkv_b = lv("attn.in_proj_bias");
            L.o = lm("attn.out_proj.weight");
            L.o_b = lv("attn.out_proj.bias");
            L.ln2g = lv("ln_2.weight");
            L.ln2b = lv("ln_2.bias");
            L.fc1 = lm("mlp.c_fc.weight");
            L.fc1_b = lv("mlp.c_fc.bias");
            L.fc2 = lm("mlp.c_proj.weight");
            L.fc2_b = lv("mlp.c_proj.bias");
        }
        C.layers.push_back(std::move(L));
    }
    for (uint32_t i = 0; i < n; ++i) {
        ClipLayer& L = C.layers[i];
        const std::string m = std::format("lora_te{}_text_model.encoder.layers.{}.", which + 1, i);
        const uint32_t E = uint32_t(L.o.dims[0]);
        lora_[lora_key(m + "self_attn.q_proj")] = {&L.qkv, 0, E};
        lora_[lora_key(m + "self_attn.k_proj")] = {&L.qkv, E, E};
        lora_[lora_key(m + "self_attn.v_proj")] = {&L.qkv, 2 * E, E};
        lora_[lora_key(m + "self_attn.out_proj")] = {&L.o, 0, E};
        lora_[lora_key(m + "mlp.fc1")] = {&L.fc1, 0, uint32_t(L.fc1.dims[1])};
        lora_[lora_key(m + "mlp.fc2")] = {&L.fc2, 0, E};
    }
    C.E = uint32_t(C.layers.front().o.dims[0]);
    C.quick_gelu = which == 0;
    C.tok = files_.find(p + (which == 0 ? "embeddings.token_embedding.weight" : "token_embedding.weight"));
    if (!C.tok || C.tok.tensor->type != gguf::Type::F16 || C.tok.tensor->dims[0] != C.E)
        throw std::runtime_error(std::format("text encoder {}: token embeddings missing or not F16", which));
    C.pos = floats(files_, p + (which == 0 ? "embeddings.position_embedding.weight" : "positional_embedding"));
    if (C.pos.size() != uint64_t(kTokens) * C.E)
        throw std::runtime_error(std::format("text encoder {}: not 77 positions", which));
    if (which == 1) {
        C.lnf_g = floats(files_, p + "ln_final.weight");
        C.lnf_b = floats(files_, p + "ln_final.bias");
        C.proj = floats(files_, p + "text_projection");
        if (C.proj.size() != uint64_t(C.E) * kPooled)
            throw std::runtime_error("text encoder 1: the projection is not 1280 x 1280");
    }
    if (C.E % kD)
        throw std::runtime_error(std::format("text encoder {}: {} wide, not a multiple of 64", which, C.E));
}

Sdxl::Res Sdxl::load_res(const std::string& p, std::vector<std::string>& emb_w, std::vector<std::string>& emb_b)
{
    const std::string P = kUnet + p;
    auto vec = [&](const std::string& n) { return upload_f32(files_, P + n); };
    auto conv = [&](const std::string& n, uint32_t k) {
        const std::string name = P + n;
        return load_conv(files_, std::span(&name, 1), k);
    };
    Res r;
    r.n1g = vec(".in_layers.0.weight");
    r.n1b = vec(".in_layers.0.bias");
    r.c1 = conv(".in_layers.2", 3);
    r.n2g = vec(".out_layers.0.weight");
    r.n2b = vec(".out_layers.0.bias");
    r.c2 = conv(".out_layers.3", 3);
    if (files_.find(P + ".skip_connection.weight")) {
        r.skip = conv(".skip_connection", 1);
        r.has_skip = true;
    }
    r.emb_off = emb_rows_;
    emb_rows_ += r.c1.cout;
    emb_w.push_back(P + ".emb_layers.1.weight");
    emb_b.push_back(P + ".emb_layers.1.bias");
    return r;
}

Sdxl::Trans Sdxl::load_trans(const std::string& p)
{
    const std::string P = kUnet + p;
    auto mat = [&](const std::string& n) { return matrix(P + n); };
    auto vec = [&](const std::string& n) { return upload_f32(files_, P + n); };
    Trans t;
    t.ng = vec(".norm.weight");
    t.nb = vec(".norm.bias");
    t.pin = mat(".proj_in.weight");
    t.pinb = vec(".proj_in.bias");
    t.pout = mat(".proj_out.weight");
    t.poutb = vec(".proj_out.bias");
    t.c = uint32_t(t.pin.dims[1]);
    t.first = n_tblocks_;
    if (t.pin.dims.size() != 2 || t.c % kD)
        throw std::runtime_error(p + ": proj_in is not a linear layer of a multiple of 64");
    for (uint32_t i = 0; files_.find(std::format("{}.transformer_blocks.{}.norm1.weight", P, i)); ++i) {
        const std::string b = std::format(".transformer_blocks.{}.", i);
        TBlock B;
        B.n1g = vec(b + "norm1.weight");
        B.n1b = vec(b + "norm1.bias");
        B.n2g = vec(b + "norm2.weight");
        B.n2b = vec(b + "norm2.bias");
        B.n3g = vec(b + "norm3.weight");
        B.n3b = vec(b + "norm3.bias");
        const std::string qkv[3] = {P + b + "attn1.to_q.weight", P + b + "attn1.to_k.weight",
                                    P + b + "attn1.to_v.weight"};
        B.qkv = upload_rows(files_, qkv);
        B.o1 = mat(b + "attn1.to_out.0.weight");
        B.o1b = vec(b + "attn1.to_out.0.bias");
        B.q2 = mat(b + "attn2.to_q.weight");
        const std::string kv[2] = {P + b + "attn2.to_k.weight", P + b + "attn2.to_v.weight"};
        B.kv2 = upload_rows(files_, kv);
        if (B.kv2.dims[0] != kCtxWidth)
            throw std::runtime_error(p + ": cross-attention does not read 2048-wide text");
        B.o2 = mat(b + "attn2.to_out.0.weight");
        B.o2b = vec(b + "attn2.to_out.0.bias");
        B.ff1 = mat(b + "ff.net.0.proj.weight");
        B.ff1b = vec(b + "ff.net.0.proj.bias");
        B.ff2 = mat(b + "ff.net.2.weight");
        B.ff2b = vec(b + "ff.net.2.bias");
        t.blocks.push_back(std::move(B));
        ++n_tblocks_;
    }
    return t;
}

void Sdxl::load_unet()
{
    auto mat = [&](const std::string& n) { return matrix(kUnet + n); };
    auto vec = [&](const std::string& n) { return upload_f32(files_, kUnet + n); };
    auto has = [&](const std::string& n) { return bool(files_.find(kUnet + n)); };
    auto conv = [&](const std::string& n, uint32_t k) {
        const std::string name = kUnet + n;
        return load_conv(files_, std::span(&name, 1), k);
    };
    std::vector<std::string> emb_w, emb_b;
    for (uint32_t i = 0;; ++i) {
        const std::string p = std::format("input_blocks.{}", i);
        Block b;
        if (has(p + ".0.weight")) {
            b.kind = Block::conv_in;
            b.conv = conv(p + ".0", 3);
        } else if (has(p + ".0.op.weight")) {
            b.kind = Block::down;
            b.conv = conv(p + ".0.op", 3);
            b.conv.stride = 2;
        } else if (has(p + ".0.in_layers.0.weight")) {
            b.r = load_res(p + ".0", emb_w, emb_b);
        } else {
            break;
        }
        if (has(p + ".1.norm.weight")) {
            b.t = load_trans(p + ".1");
            b.has_t = true;
        }
        in_.push_back(std::move(b));
    }
    mid0_ = load_res("middle_block.0", emb_w, emb_b);
    mid_t_ = load_trans("middle_block.1");
    mid1_ = load_res("middle_block.2", emb_w, emb_b);
    for (uint32_t i = 0; has(std::format("output_blocks.{}.0.in_layers.0.weight", i)); ++i) {
        const std::string p = std::format("output_blocks.{}", i);
        Block b;
        b.r = load_res(p + ".0", emb_w, emb_b);
        if (has(p + ".1.norm.weight")) {
            b.t = load_trans(p + ".1");
            b.has_t = true;
        }
        for (const char* u : {".1.conv", ".2.conv"})
            if (has(p + u + ".weight")) {
                b.conv = conv(p + u, 3);
                b.up = true;
            }
        out_.push_back(std::move(b));
    }
    if (in_.empty() || in_.front().kind != Block::conv_in || out_.size() != in_.size())
        throw std::runtime_error("UNet: not SDXL's input and output blocks");
    emb_w_ = upload_rows(files_, emb_w);
    {
        std::vector<float> b;
        for (const auto& n : emb_b) {
            const auto v = floats(files_, n);
            b.insert(b.end(), v.begin(), v.end());
        }
        emb_b_ = reserve(b.size() * 4, gguf::Type::F32, {b.size()});
        std::memcpy(static_cast<char*>(emb_b_.buf->contents()) + emb_b_.off, b.data(), b.size() * 4);
    }
    t1_ = mat("time_embed.0.weight");
    t1b_ = vec("time_embed.0.bias");
    t2_ = mat("time_embed.2.weight");
    t2b_ = vec("time_embed.2.bias");
    l1_ = mat("label_emb.0.0.weight");
    l1b_ = vec("label_emb.0.0.bias");
    l2_ = mat("label_emb.0.2.weight");
    l2b_ = vec("label_emb.0.2.bias");
    if (l1_.dims[0] != kAdded)
        throw std::runtime_error("UNet: the added conditioning is not 2816 wide");
    out_g_ = vec("out.0.weight");
    out_b_ = vec("out.0.bias");
    out_c_ = conv("out.2", 3);
    auto targets = [&](const std::string& p, Trans& t) {
        lora_[lora_key(p + ".proj_in")] = {&t.pin, 0, t.c};
        lora_[lora_key(p + ".proj_out")] = {&t.pout, 0, t.c};
        for (uint32_t i = 0; i < t.blocks.size(); ++i) {
            TBlock& B = t.blocks[i];
            const std::string b = lora_key(std::format("{}.transformer_blocks.{}.", p, i));
            const uint32_t c = t.c;
            lora_[b + "attn1.to.q"] = {&B.qkv, 0, c};
            lora_[b + "attn1.to.k"] = {&B.qkv, c, c};
            lora_[b + "attn1.to.v"] = {&B.qkv, 2 * c, c};
            lora_[b + "attn1.to.out.0"] = {&B.o1, 0, c};
            lora_[b + "attn2.to.q"] = {&B.q2, 0, c};
            lora_[b + "attn2.to.k"] = {&B.kv2, 0, c};
            lora_[b + "attn2.to.v"] = {&B.kv2, c, c};
            lora_[b + "attn2.to.out.0"] = {&B.o2, 0, c};
            lora_[b + "ff.net.0.proj"] = {&B.ff1, 0, uint32_t(B.ff1.dims[1])};
            lora_[b + "ff.net.2"] = {&B.ff2, 0, c};
        }
    };
    for (uint32_t i = 0; i < in_.size(); ++i)
        if (in_[i].has_t)
            targets(std::format("input_blocks.{}.1", i), in_[i].t);
    targets("middle_block.1", mid_t_);
    for (uint32_t i = 0; i < out_.size(); ++i)
        if (out_[i].has_t)
            targets(std::format("output_blocks.{}.1", i), out_[i].t);
}

Engine::LoraTarget Sdxl::lora_target(const std::string& module)
{
    const auto it = lora_.find(lora_key(module));
    return it == lora_.end() ? LoraTarget{} : it->second;
}

void Sdxl::lora_merged()
{
    empty_[0] = encode_clip(0, empty_ids(kPadL));
    empty_[1] = encode_clip(1, empty_ids(kPadG));
}

void Sdxl::ln(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* x, const Tensor& g, const Tensor& b, MTL::Buffer* y,
              uint32_t rows, uint32_t n)
{
    const LnArgs a{n, 1e-5f};
    enc->setComputePipelineState(pso("layer_norm"));
    enc->setBuffer(x, 0, 0);
    enc->setBuffer(g.buf, g.off, 1);
    enc->setBuffer(b.buf, b.off, 2);
    enc->setBuffer(y, 0, 3);
    enc->setBytes(&a, sizeof(a), 4);
    enc->dispatchThreadgroups(MTL::Size(rows, 1, 1), MTL::Size(256, 1, 1));
    after(enc, "layer_norm");
}

void Sdxl::heads64(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* x, uint32_t ld, uint32_t col0, uint32_t n_heads,
                   uint32_t T, uint32_t n_pad, MTL::Buffer* y)
{
    const HeadsArgs a{T, ld, col0, kD, n_pad, 0, 0};
    dispatch(enc, "heads_half", {x, x, y}, &a, sizeof(a), MTL::Size(kD / 2, T, n_heads), MTL::Size(32, 1, 1));
    if (n_pad > T) {
        const uint32_t c[4] = {kD, n_pad, T, 0};
        dispatch(enc, "heads_clear", {y}, c, sizeof(c), MTL::Size(kD, n_pad - T, n_heads), MTL::Size(64, 1, 1));
    }
}

void Sdxl::attend64(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* q, MTL::Buffer* k, MTL::Buffer* v, MTL::Buffer* o,
                    uint32_t n_q, uint32_t n_k, uint32_t n_heads, uint32_t block)
{
    const uint32_t q_pad = round_up(n_q, kQRows), k_pad = round_up(n_k, 128);
    const AttnArgs a{n_q, q_pad, n_k, k_pad, n_heads, n_heads, 0, block, n_heads * kD, 1.0f / std::sqrt(float(kD))};
    dispatch(enc, "flash_attn_sg_64_16x32_4", {q, k, v, o}, &a, sizeof(a), MTL::Size(q_pad / kQRows, n_heads, 1),
             MTL::Size(128, 1, 1), false);
}

std::vector<float> Sdxl::encode_clip(int which, std::span<const int32_t> ids, std::vector<float>* pooled)
{
    const Clip& C = clip_[size_t(which)];
    const uint32_t T = uint32_t(ids.size()), E = C.E, H = E / kD, nff = uint32_t(C.layers.front().fc1.dims[1]);
    if (T != kTokens)
        throw std::runtime_error("a text encoder's chunk is 77 tokens");
    const uint32_t q_pad = round_up(T, kQRows), k_pad = round_up(T, 128);
    MTL::Buffer* x = scratch("c_x", uint64_t(T) * E * 4);
    MTL::Buffer* xn = scratch("c_xn", uint64_t(T) * E * 4);
    MTL::Buffer* qkv = scratch("c_qkv", uint64_t(T) * 3 * E * 4);
    MTL::Buffer* qh = scratch("c_qh", uint64_t(H) * q_pad * kD * 2);
    MTL::Buffer* kh = scratch("c_kh", uint64_t(H) * k_pad * kD * 2);
    MTL::Buffer* vh = scratch("c_vh", uint64_t(H) * k_pad * kD * 2);
    MTL::Buffer* att = scratch("c_att", uint64_t(T) * E * 4);
    MTL::Buffer* y = scratch("c_y", uint64_t(T) * E * 4);
    MTL::Buffer* ff = scratch("c_ff", uint64_t(T) * nff * 4);
    MTL::Buffer* hidden = scratch("c_hidden", uint64_t(T) * E * 4);
    {
        auto* xf = static_cast<float*>(x->contents());
        const uint64_t vocab = C.tok.tensor->dims[1];
        std::vector<__fp16> row(E);
        for (uint32_t t = 0; t < T; ++t) {
            if (ids[t] < 0 || uint64_t(ids[t]) >= vocab)
                throw std::runtime_error("token out of range");
            gguf::read_exact(C.tok.file->fd(), row.data(), uint64_t(E) * 2,
                             C.tok.tensor->offset + uint64_t(ids[t]) * E * 2);
            for (uint32_t i = 0; i < E; ++i)
                xf[uint64_t(t) * E + i] = float(row[i]) + C.pos[uint64_t(t) * E + i];
        }
    }
    MTL::CommandBuffer* cb = ctx_.queue()->commandBuffer();
    cur_cb_ = cb;
    MTL::ComputeCommandEncoder* enc = encoder(cb, 0, -1);
    const uint32_t n_layers = pooled ? uint32_t(C.layers.size()) : C.layers_total - 1;
    for (uint32_t il = 0; il < n_layers; ++il) {
        const ClipLayer& L = C.layers[il];
        ln(enc, x, L.ln1g, L.ln1b, xn, T, E);
        proj(enc, L.qkv, xn, 0, qkv, 0, T);
        bias(enc, qkv, L.qkv_b, 3 * E, T);
        heads64(enc, qkv, 3 * E, 0, H, T, q_pad, qh);
        heads64(enc, qkv, 3 * E, E, H, T, k_pad, kh);
        heads64(enc, qkv, 3 * E, 2 * E, H, T, k_pad, vh);
        attend64(enc, qh, kh, vh, att, T, T, H, UINT32_MAX);
        proj(enc, L.o, att, 0, y, 0, T);
        bias(enc, y, L.o_b, E, T);
        add(enc, x, y, x, T * E);
        ln(enc, x, L.ln2g, L.ln2b, xn, T, E);
        proj(enc, L.fc1, xn, 0, ff, 0, T);
        bias(enc, ff, L.fc1_b, nff, T);
        unary(enc, C.quick_gelu ? "quick_gelu" : "gelu_erf", ff, T * nff);
        proj(enc, L.fc2, ff, 0, y, 0, T);
        bias(enc, y, L.fc2_b, E, T);
        add(enc, x, y, x, T * E);
        if (il + 2 == C.layers_total) {
            const uint32_t n = T * E;
            dispatch(enc, "copy_f32", {x, hidden}, &n, sizeof(n), MTL::Size(n, 1, 1), MTL::Size(256, 1, 1));
        }
    }
    enc->endEncoding();
    finish(cb);
    const auto* hs = static_cast<const float*>(hidden->contents());
    std::vector<float> out(hs, hs + uint64_t(T) * E);
    if (pooled) {
        // The final state at the first end token, layer normed, times the projection ([E][1280] as stored).
        const uint32_t eos = uint32_t(std::find(ids.begin(), ids.end(), ClipTokenizer::kEnd) - ids.begin());
        if (eos >= T)
            throw std::runtime_error("a chunk without its end token");
        const float* r = static_cast<const float*>(x->contents()) + uint64_t(eos) * E;
        double mean = 0, var = 0;
        for (uint32_t i = 0; i < E; ++i)
            mean += r[i];
        mean /= E;
        for (uint32_t i = 0; i < E; ++i)
            var += (r[i] - mean) * (r[i] - mean);
        const double inv = 1.0 / std::sqrt(var / E + 1e-5);
        std::vector<double> nr(E);
        for (uint32_t i = 0; i < E; ++i)
            nr[i] = (r[i] - mean) * inv * C.lnf_g[i] + C.lnf_b[i];
        pooled->assign(kPooled, 0.0f);
        for (uint32_t j = 0; j < kPooled; ++j) {
            double s = 0;
            for (uint32_t i = 0; i < E; ++i)
                s += nr[i] * C.proj[uint64_t(i) * kPooled + j];
            (*pooled)[j] = float(s);
        }
    }
    return out;
}

std::vector<std::vector<Sdxl::Token>> Sdxl::chunks(const std::string& prompt, int32_t pad) const
{
    // Words: the weighted parts, also split before "embedding:" after whitespace (where a textual inversion would
    // stand; none is read, the text is tokenized as it is).
    std::vector<std::vector<Token>> words;
    for (const auto& [part, w] : prompt_weights(prompt)) {
        std::vector<std::string> split{""};
        for (size_t i = 0; i < part.size(); ++i) {
            if (i > 0 && std::isspace(static_cast<unsigned char>(part[i - 1])) &&
                std::string_view(part).substr(i).starts_with("embedding:"))
                split.emplace_back();
            split.back() += part[i];
        }
        for (const auto& s : split) {
            if (s.empty())
                continue;
            std::vector<Token> t;
            for (int32_t id : clip_tok_->encode(s))
                t.push_back({id, w});
            words.push_back(std::move(t));
        }
    }
    constexpr size_t kMax = kTokens, kLong = 8;
    std::vector<std::vector<Token>> out(1, {{ClipTokenizer::kStart, 1.0f}});
    for (auto& word : words) {
        const bool large = word.size() >= kLong;
        std::span<const Token> rest(word);
        while (!rest.empty()) {
            auto& batch = out.back();
            if (rest.size() + batch.size() > kMax - 1) {
                const size_t room = kMax - batch.size() - 1;
                if (large) {
                    batch.insert(batch.end(), rest.begin(), rest.begin() + ptrdiff_t(room));
                    batch.push_back({ClipTokenizer::kEnd, 1.0f});
                    rest = rest.subspan(room);
                } else {
                    batch.push_back({ClipTokenizer::kEnd, 1.0f});
                    batch.insert(batch.end(), room, {pad, 1.0f});
                }
                out.push_back({{ClipTokenizer::kStart, 1.0f}});
            } else {
                batch.insert(batch.end(), rest.begin(), rest.end());
                rest = {};
            }
        }
    }
    out.back().push_back({ClipTokenizer::kEnd, 1.0f});
    out.back().resize(kMax, {pad, 1.0f});
    return out;
}

std::vector<float> Sdxl::prompt_embeddings(const std::string& prompt)
{
    const auto cl = chunks(prompt, kPadL), cg = chunks(prompt, kPadG);
    bool weighted = false;
    for (const auto& c : cl)
        for (const Token& t : c)
            weighted = weighted || t.weight != 1.0f;
    const uint32_t S = uint32_t(cl.size()) * kTokens, EL = clip_[0].E, EG = clip_[1].E;
    std::vector<float> out{float(S)};
    out.resize(1 + uint64_t(S) * kCtxWidth);
    std::vector<float> pooled;
    for (size_t k = 0; k < cl.size(); ++k) {
        std::vector<int32_t> il, ig;
        for (const Token& t : cl[k])
            il.push_back(t.id);
        for (const Token& t : cg[k])
            ig.push_back(t.id);
        auto zl = encode_clip(0, il);
        auto zg = encode_clip(1, ig, k == 0 ? &pooled : nullptr);
        // A weighted token's states move away from the empty prompt's by its weight.
        if (weighted)
            for (uint32_t t = 0; t < kTokens; ++t) {
                const float w = cl[k][t].weight;
                if (w == 1.0f)
                    continue;
                for (uint32_t i = 0; i < EL; ++i) {
                    const float e = empty_[0][uint64_t(t) * EL + i];
                    zl[uint64_t(t) * EL + i] = (zl[uint64_t(t) * EL + i] - e) * w + e;
                }
                for (uint32_t i = 0; i < EG; ++i) {
                    const float e = empty_[1][uint64_t(t) * EG + i];
                    zg[uint64_t(t) * EG + i] = (zg[uint64_t(t) * EG + i] - e) * w + e;
                }
            }
        for (uint32_t t = 0; t < kTokens; ++t) {
            float* row = out.data() + 1 + (k * kTokens + t) * uint64_t(kCtxWidth);
            std::memcpy(row, zl.data() + uint64_t(t) * EL, EL * 4);
            std::memcpy(row + EL, zg.data() + uint64_t(t) * EG, EG * 4);
        }
    }
    out.insert(out.end(), pooled.begin(), pooled.end());
    return out;
}

uint32_t Sdxl::timestep(float sigma) const
{
    const float ls = std::log(sigma);
    uint32_t best = 0;
    for (uint32_t i = 1; i < 1000; ++i)
        if (std::abs(ls - log_sigma_[i]) < std::abs(ls - log_sigma_[best]))
            best = i;
    return best;
}

std::vector<float> Sdxl::sigmas(uint32_t steps, uint32_t, std::optional<float>) const
{
    if (!steps)
        throw std::runtime_error("no steps");
    // Timesteps from 999 to 0 as a float linspace computes them (from both ends towards the middle).
    std::vector<float> s;
    const float start = 999.0f, end = 0.0f;
    const float stepv = steps > 1 ? (end - start) / float(steps - 1) : 0.0f;
    for (uint32_t i = 0; i < steps; ++i) {
        const float t = i < steps / 2 ? start + stepv * float(i) : end - stepv * float(steps - 1 - i);
        const float tc = std::clamp(t, 0.0f, 999.0f);
        const uint32_t lo = uint32_t(std::floor(tc)), hi = uint32_t(std::ceil(tc));
        const float w = tc - std::floor(tc);
        s.push_back(std::exp((1.0f - w) * log_sigma_[lo] + w * log_sigma_[hi]));
    }
    s.push_back(0.0f);
    return s;
}

void Sdxl::begin(std::span<const float> prompt, uint32_t h, uint32_t w, uint32_t slot)
{
    if (slot > 1)
        throw std::runtime_error("slot must be 0 or 1");
    if (prompt.empty())
        throw std::runtime_error("prompt embeddings: no header");
    const uint32_t S = uint32_t(prompt[0]);
    if (!S || prompt.size() != 1 + uint64_t(S) * kCtxWidth + kPooled)
        throw std::runtime_error("prompt embeddings: sizes do not add up");
    gh_ = h;
    gw_ = w;
    ctx_rows_[slot] = S;
    // The added conditioning: the pooled state, then the image's height, width, crop (0, 0) and target size (the
    // same), 256 sinusoid values each.
    auto& y = y_[slot];
    y.assign(kAdded, 0.0f);
    std::copy(prompt.end() - kPooled, prompt.end(), y.begin());
    const float H = float(16 * h), W = float(16 * w);
    const float sizes[6] = {H, W, 0.0f, 0.0f, H, W};
    for (uint32_t i = 0; i < 6; ++i)
        sinusoid(sizes[i], 256, y.data() + kPooled + i * 256);
    // Each transformer block's keys and values of the context.
    const uint32_t s_pad = round_up(S, 128);
    MTL::Buffer* ctx = scratch("x_ctx", uint64_t(S) * kCtxWidth * 4);
    std::memcpy(ctx->contents(), prompt.data() + 1, uint64_t(S) * kCtxWidth * 4);
    MTL::Buffer* kv = scratch("x_kv", uint64_t(S) * 2 * 1280 * 4);
    MTL::CommandBuffer* cb = ctx_.queue()->commandBuffer();
    cur_cb_ = cb;
    MTL::ComputeCommandEncoder* enc = encoder(cb, 0, -1);
    auto each = [&](const Trans& t) {
        for (uint32_t b = 0; b < t.blocks.size(); ++b) {
            const uint32_t c = t.c, H = c / kD, idx = t.first + b;
            MTL::Buffer* k = scratch(std::format("x_k{}_{}", slot, idx), uint64_t(H) * s_pad * kD * 2);
            MTL::Buffer* v = scratch(std::format("x_v{}_{}", slot, idx), uint64_t(H) * s_pad * kD * 2);
            proj(enc, t.blocks[b].kv2, ctx, 0, kv, 0, S);
            heads64(enc, kv, 2 * c, 0, H, S, s_pad, k);
            heads64(enc, kv, 2 * c, c, H, S, s_pad, v);
        }
    };
    for (const Block& b : in_)
        if (b.has_t)
            each(b.t);
    each(mid_t_);
    for (const Block& b : out_)
        if (b.has_t)
            each(b.t);
    enc->endEncoding();
    finish(cb);
}

void Sdxl::res_fwd(MTL::ComputeCommandEncoder*& enc, const Res& r, MTL::Buffer* x, uint32_t h, uint32_t w,
                   MTL::Buffer* y, MTL::Buffer* emb)
{
    const uint32_t rows = h * w, cin = r.c1.cin, cout = r.c2.cout;
    MTL::Buffer* t1 = scratch("u_t1", uint64_t(rows) * std::max(cin, cout) * 4);
    MTL::Buffer* t2 = scratch("u_t2", uint64_t(rows) * cout * 4);
    group_norm(enc, x, r.n1g, r.n1b, t1, rows, cin, true, 1e-5f);
    conv(enc, r.c1, t1, h, w, t2);
    const Tensor e{emb, uint64_t(r.emb_off) * 4, gguf::Type::F32, {cout}};
    bias(enc, t2, e, cout, rows);
    group_norm(enc, t2, r.n2g, r.n2b, t2, rows, cout, true, 1e-5f);
    conv(enc, r.c2, t2, h, w, y);
    if (r.has_skip) {
        conv(enc, r.skip, x, h, w, t1);
        add(enc, y, t1, y, rows * cout);
    } else {
        add(enc, y, x, y, rows * cout);
    }
}

void Sdxl::trans_fwd(MTL::ComputeCommandEncoder*& enc, const Trans& t, MTL::Buffer* x, uint32_t n, MTL::Buffer* y,
                     uint32_t slot)
{
    const uint32_t c = t.c, H = c / kD;
    // The GEMMs run a multiple of 256 rows (the rest ignored), which keeps them on their largest tiles.
    const uint32_t Ng = n > 256 ? round_up(n, 256) : n;
    const uint32_t nff = uint32_t(t.blocks.front().ff1.dims[1]), inner = nff / 2;
    const uint32_t q_pad = round_up(n, kQRows), k_pad = round_up(n, 128);
    const uint32_t S = ctx_rows_[slot], s_pad = round_up(S, 128);
    MTL::Buffer* xn = scratch("t_xn", uint64_t(Ng) * c * 4);
    MTL::Buffer* hs = scratch("t_h", uint64_t(Ng) * c * 4);
    MTL::Buffer* hn = scratch("t_hn", uint64_t(Ng) * c * 4);
    MTL::Buffer* qkv = scratch("t_qkv", uint64_t(Ng) * 3 * c * 4);
    MTL::Buffer* qh = scratch("t_qh", uint64_t(H) * q_pad * kD * 2);
    MTL::Buffer* kh = scratch("t_kh", uint64_t(H) * k_pad * kD * 2);
    MTL::Buffer* vh = scratch("t_vh", uint64_t(H) * k_pad * kD * 2);
    MTL::Buffer* att = scratch("t_att", uint64_t(Ng) * c * 4);
    MTL::Buffer* o = scratch("t_o", uint64_t(Ng) * c * 4);
    MTL::Buffer* ff = scratch("t_ff", uint64_t(Ng) * nff * 4);
    MTL::Buffer* m = scratch("t_m", uint64_t(Ng) * inner * 4);
    group_norm(enc, x, t.ng, t.nb, xn, n, c, false, 1e-6f);
    proj(enc, t.pin, xn, 0, hs, 0, Ng);
    bias(enc, hs, t.pinb, c, n);
    for (uint32_t b = 0; b < t.blocks.size(); ++b) {
        const TBlock& B = t.blocks[b];
        cur_layer_ = t.first + b;
        ln(enc, hs, B.n1g, B.n1b, hn, n, c);
        proj(enc, B.qkv, hn, 0, qkv, 0, Ng);
        heads64(enc, qkv, 3 * c, 0, H, n, q_pad, qh);
        heads64(enc, qkv, 3 * c, c, H, n, k_pad, kh);
        heads64(enc, qkv, 3 * c, 2 * c, H, n, k_pad, vh);
        attend64(enc, qh, kh, vh, att, n, n, H, 0);
        proj(enc, B.o1, att, 0, o, 0, Ng);
        bias(enc, o, B.o1b, c, n);
        add(enc, hs, o, hs, n * c);
        ln(enc, hs, B.n2g, B.n2b, hn, n, c);
        proj(enc, B.q2, hn, 0, qkv, 0, Ng);
        heads64(enc, qkv, c, 0, H, n, q_pad, qh);
        attend64(enc, qh, scratch(std::format("x_k{}_{}", slot, t.first + b), uint64_t(H) * s_pad * kD * 2),
                 scratch(std::format("x_v{}_{}", slot, t.first + b), uint64_t(H) * s_pad * kD * 2), att, n, S, H, 0);
        proj(enc, B.o2, att, 0, o, 0, Ng);
        bias(enc, o, B.o2b, c, n);
        add(enc, hs, o, hs, n * c);
        ln(enc, hs, B.n3g, B.n3b, hn, n, c);
        proj(enc, B.ff1, hn, 0, ff, 0, Ng);
        bias(enc, ff, B.ff1b, nff, n);
        dispatch(enc, "geglu_rows", {ff, m}, &inner, sizeof(inner), MTL::Size(inner, n, 1), MTL::Size(256, 1, 1));
        proj(enc, B.ff2, m, 0, o, 0, Ng);
        bias(enc, o, B.ff2b, c, n);
        add(enc, hs, o, hs, n * c);
    }
    proj(enc, t.pout, hs, 0, o, 0, Ng);
    bias(enc, o, t.poutb, c, n);
    add(enc, o, x, y, n * c);
}

std::vector<float> Sdxl::step(std::span<const float> latents, float sigma, uint32_t slot)
{
    if (slot > 1)
        throw std::runtime_error("slot must be 0 or 1");
    const uint32_t N = gh_ * gw_, lh = 2 * gh_, lw = 2 * gw_, R = lh * lw;
    if (latents.size() != uint64_t(N) * 16)
        throw std::runtime_error("latents must be [h * w][16]");
    if (y_[slot].empty())
        throw std::runtime_error("step before begin");
    // The widest image of a level (rows times channels): a block's concatenated input, or the GEGLU's.
    uint64_t most = uint64_t(R) * 4;
    {
        uint32_t rows = R;
        for (const Block& b : in_) {
            if (b.kind == Block::down)
                rows /= 4;
            most = std::max(most, uint64_t(rows) * std::max(b.conv.cout, b.r.c2.cout));
        }
        for (const Block& b : out_) {
            most = std::max(most, uint64_t(rows) * std::max(b.r.c1.cin, b.r.c2.cout));
            if (b.up)
                rows *= 4;
        }
    }
    MTL::Buffer* bufs[3] = {scratch("u_a", most * 4), scratch("u_b", most * 4), scratch("u_c", most * 4)};
    MTL::Buffer* cat = scratch("u_cat", most * 4);
    MTL::Buffer* upb = scratch("u_up", most * 4);
    MTL::Buffer* lat = scratch("u_lat", uint64_t(R) * 4 * 4);
    MTL::Buffer* te = scratch("u_te", 320 * 4);
    MTL::Buffer* yv = scratch("u_y", kAdded * 4);
    MTL::Buffer* e1 = scratch("u_e1", 1280 * 4);
    MTL::Buffer* emb = scratch("u_emb", 1280 * 4);
    MTL::Buffer* l1 = scratch("u_l1", 1280 * 4);
    MTL::Buffer* l2 = scratch("u_l2", 1280 * 4);
    MTL::Buffer* eo = scratch("u_eo", uint64_t(emb_rows_) * 4);
    MTL::Buffer* eps = scratch("u_eps", uint64_t(R) * 4 * 4);
    unpack(latents, gh_, gw_, 1.0f / std::sqrt(sigma * sigma + 1.0f), static_cast<float*>(lat->contents()));
    sinusoid(float(timestep(sigma)), 320, static_cast<float*>(te->contents()));
    std::memcpy(yv->contents(), y_[slot].data(), kAdded * 4);

    MTL::CommandBuffer* cb = ctx_.queue()->commandBuffer();
    cur_cb_ = cb;
    MTL::ComputeCommandEncoder* enc = encoder(cb, 0, -1);
    // The timestep's embedding plus the added conditioning's, then every residual block's projection of it.
    proj(enc, t1_, te, 0, e1, 0, 1);
    bias(enc, e1, t1b_, 1280, 1);
    unary(enc, "silu", e1, 1280);
    proj(enc, t2_, e1, 0, emb, 0, 1);
    bias(enc, emb, t2b_, 1280, 1);
    probe(enc, "temb", -1, emb, 1280);
    proj(enc, l1_, yv, 0, l1, 0, 1);
    bias(enc, l1, l1b_, 1280, 1);
    unary(enc, "silu", l1, 1280);
    proj(enc, l2_, l1, 0, l2, 0, 1);
    bias(enc, l2, l2b_, 1280, 1);
    probe(enc, "yemb", -1, l2, 1280);
    add(enc, emb, l2, emb, 1280);
    unary(enc, "silu", emb, 1280);
    proj(enc, emb_w_, emb, 0, eo, 0, 1);
    bias(enc, eo, emb_b_, emb_rows_, 1);

    // Each input block's output stays for its output block.
    std::vector<MTL::Buffer*> skips;
    std::vector<uint32_t> skip_c;
    uint32_t h = lh, w = lw;
    MTL::Buffer* cur = lat;
    for (uint32_t i = 0; i < in_.size(); ++i) {
        const Block& b = in_[i];
        uint32_t c = 0;
        if (b.kind != Block::res) {
            c = b.conv.cout;
            const uint32_t oh = b.kind == Block::down ? h / 2 : h, ow = b.kind == Block::down ? w / 2 : w;
            MTL::Buffer* s = scratch(std::format("u_s{}", i), uint64_t(oh) * ow * c * 4);
            conv(enc, b.conv, cur, h, w, s);
            h = oh;
            w = ow;
            cur = s;
        } else {
            c = b.r.c2.cout;
            MTL::Buffer* s = scratch(std::format("u_s{}", i), uint64_t(h) * w * c * 4);
            if (b.has_t) {
                res_fwd(enc, b.r, cur, h, w, bufs[0], eo);
                trans_fwd(enc, b.t, bufs[0], h * w, s, slot);
            } else {
                res_fwd(enc, b.r, cur, h, w, s, eo);
            }
            cur = s;
        }
        probe(enc, "in", int(i), cur, uint64_t(h) * w * c);
        skips.push_back(cur);
        skip_c.push_back(c);
    }
    res_fwd(enc, mid0_, cur, h, w, bufs[0], eo);
    trans_fwd(enc, mid_t_, bufs[0], h * w, bufs[1], slot);
    res_fwd(enc, mid1_, bufs[1], h, w, bufs[2], eo);
    cur = bufs[2];
    uint32_t c = mid1_.c2.cout;
    probe(enc, "mid", -1, cur, uint64_t(h) * w * c);
    // Free buffers among the three, other than cur.
    auto other = [&](MTL::Buffer* not1, MTL::Buffer* not2) {
        for (MTL::Buffer* b : bufs)
            if (b != not1 && b != not2)
                return b;
        return bufs[0];
    };
    for (uint32_t i = 0; i < out_.size(); ++i) {
        const Block& b = out_[i];
        MTL::Buffer* skip = skips.back();
        const uint32_t sc = skip_c.back(), rows = h * w;
        skips.pop_back();
        skip_c.pop_back();
        // [cur | skip] per pixel.
        for (const auto& [src, off, n] : {std::tuple{cur, 0u, c}, std::tuple{skip, c, sc}}) {
            const uint32_t a[2] = {n, c + sc};
            enc->setComputePipelineState(pso("copy_rows"));
            enc->setBuffer(src, 0, 0);
            enc->setBuffer(cat, uint64_t(off) * 4, 1);
            enc->setBytes(a, sizeof(a), 2);
            enc->dispatchThreads(MTL::Size(n, rows, 1), MTL::Size(256, 1, 1));
            after(enc, "copy_rows");
        }
        MTL::Buffer* r = other(cur, nullptr);
        res_fwd(enc, b.r, cat, h, w, r, eo);
        cur = r;
        c = b.r.c2.cout;
        if (b.has_t) {
            MTL::Buffer* t = other(cur, nullptr);
            trans_fwd(enc, b.t, cur, rows, t, slot);
            cur = t;
        }
        if (b.up) {
            const UpArgs ua{h, w, c, c, 1, 1};
            dispatch(enc, "upsample2x", {cur, upb}, &ua, sizeof(ua), MTL::Size(c, 2 * w, 2 * h), MTL::Size(64, 1, 1));
            h *= 2;
            w *= 2;
            MTL::Buffer* u = other(cur, nullptr);
            conv(enc, b.conv, upb, h, w, u);
            cur = u;
            c = b.conv.cout;
        }
        probe(enc, "out", int(i), cur, uint64_t(h) * w * c);
    }
    MTL::Buffer* t = other(cur, nullptr);
    group_norm(enc, cur, out_g_, out_b_, t, h * w, c, true, 1e-5f);
    conv(enc, out_c_, t, h, w, eps);
    enc->endEncoding();
    finish(cb);
    // The predicted noise as tokens.
    const auto* e = static_cast<const float*>(eps->contents());
    std::vector<float> v(uint64_t(N) * 16);
    for (uint32_t y = 0; y < lh; ++y)
        for (uint32_t x = 0; x < lw; ++x)
            for (uint32_t ch = 0; ch < 4; ++ch)
                v[(uint64_t(y / 2) * gw_ + x / 2) * 16 + ch * 4 + (y % 2) * 2 + x % 2] = e[(uint64_t(y) * lw + x) * 4 + ch];
    return v;
}

} // namespace mako::image
