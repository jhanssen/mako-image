#include "image/png.hpp"

#include <ImageIO/ImageIO.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <format>
#include <fstream>
#include <stdexcept>

namespace mako::image {

namespace {

template <typename T> struct Cf {
    T ref = nullptr;
    explicit Cf(T r) : ref(r) {}
    ~Cf()
    {
        if (ref)
            CFRelease(ref);
    }
    Cf(const Cf&) = delete;
    Cf& operator=(const Cf&) = delete;
    Cf(Cf&& o) noexcept : ref(o.ref) { o.ref = nullptr; }
};

uint32_t crc32(const uint8_t* p, size_t n, uint32_t c = 0xFFFFFFFFu)
{
    static const auto table = [] {
        std::array<uint32_t, 256> t{};
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t v = i;
            for (int k = 0; k < 8; ++k)
                v = v & 1 ? 0xEDB88320u ^ (v >> 1) : v >> 1;
            t[i] = v;
        }
        return t;
    }();
    for (size_t i = 0; i < n; ++i)
        c = table[(c ^ p[i]) & 0xFF] ^ (c >> 8);
    return c;
}

// An iTXt chunk: keyword, uncompressed, no language, UTF-8 text.
std::vector<uint8_t> itxt(const std::string& key, const std::string& text)
{
    std::vector<uint8_t> data(key.begin(), key.end());
    for (uint8_t b : {0, 0, 0, 0, 0})
        data.push_back(b);
    data.insert(data.end(), text.begin(), text.end());
    std::vector<uint8_t> chunk;
    const uint32_t n = uint32_t(data.size());
    for (int s = 24; s >= 0; s -= 8)
        chunk.push_back(uint8_t(n >> s));
    const size_t type = chunk.size();
    for (char c : std::string("iTXt"))
        chunk.push_back(uint8_t(c));
    chunk.insert(chunk.end(), data.begin(), data.end());
    const uint32_t crc = ~crc32(chunk.data() + type, chunk.size() - type);
    for (int s = 24; s >= 0; s -= 8)
        chunk.push_back(uint8_t(crc >> s));
    return chunk;
}

std::string json_string(const std::string& s)
{
    std::string o = "\"";
    for (unsigned char c : s) {
        if (c == '"' || c == '\\')
            o += {'\\', char(c)};
        else if (c == '\n')
            o += "\\n";
        else if (c < 0x20)
            o += std::format("\\u{:04x}", c);
        else
            o += char(c);
    }
    return o + "\"";
}

} // namespace

std::vector<uint8_t> with_provenance(std::vector<uint8_t> png, const Provenance& p)
{
    // IEND: the last 12 bytes (length 0, type, CRC).
    if (png.size() < 20 || std::memcmp(png.data() + png.size() - 8, "IEND", 4) != 0)
        throw std::runtime_error("with_provenance: not a PNG ending in IEND");
    std::string params = p.prompt;
    if (p.negative)
        params += "\nNegative prompt: " + *p.negative;
    params += std::format("\nSteps: {}, Sampler: {}, CFG scale: {:g}, Seed: {}, Size: {}x{}, Model: {}", p.steps,
                          p.sampler == "euler_a" ? "Euler a" : p.sampler == "er_sde"      ? "ER-SDE"
                          : p.sampler == "data_space" ? "Data-space"
                                                      : "Euler", p.guidance, p.seed, p.width, p.height,
                          p.model);
    if (p.strength)
        params += std::format(", Denoising strength: {:g}", *p.strength);
    if (p.color_end)
        params += std::format(", Control color end: {:g}", *p.color_end);
    if (p.depth_end)
        params += std::format(", Control depth end: {:g}", *p.depth_end);
    std::string loras;
    for (const auto& l : p.loras)
        loras += std::format("{}{{\"name\":{},\"scale\":{:g}{}}}", loras.empty() ? "" : ",", json_string(l.name),
                             l.scale, l.merge ? std::format(",\"merge\":{}", *l.merge) : "");
    if (!p.loras.empty()) {
        params += ", Lora: ";
        for (size_t i = 0; i < p.loras.size(); ++i)
            params += std::format("{}{}:{:g}{}", i ? " " : "", p.loras[i].name, p.loras[i].scale,
                                  !p.loras[i].merge ? "" : *p.loras[i].merge ? ":merge" : ":unmerged");
    }
    std::string skip;
    for (uint32_t b : p.skip_blocks)
        skip += std::format("{}{}", skip.empty() ? "" : ",", b);
    const std::string json = std::format(
        "{{\"prompt\":{},\"negative_prompt\":{},\"model\":{},\"steps\":{},\"guidance\":{:g},\"sampler\":{},"
        "\"seed\":{},\"width\":{},\"height\":{},\"loras\":[{}]{}{}{}{}}}",
        json_string(p.prompt), p.negative ? json_string(*p.negative) : "null", json_string(p.model), p.steps,
        p.guidance, json_string(p.sampler), p.seed, p.width, p.height, loras,
        skip.empty() ? "" : std::format(",\"skip_blocks\":[{}]", skip),
        p.strength ? std::format(",\"strength\":{:g}", *p.strength) : "",
        p.color_end ? std::format(",\"control_color_end\":{:g}", *p.color_end) : "",
        p.depth_end ? std::format(",\"control_depth_end\":{:g}", *p.depth_end) : "");
    std::vector<uint8_t> chunks = itxt("parameters", params);
    const auto m = itxt("mako", json);
    chunks.insert(chunks.end(), m.begin(), m.end());
    png.insert(png.end() - 12, chunks.begin(), chunks.end());
    return png;
}

std::vector<uint8_t> to_rgba8(std::span<const float> rgba)
{
    std::vector<uint8_t> out(rgba.size());
    for (size_t i = 0; i < rgba.size(); ++i) {
        const float v = std::fmin(std::fmax(rgba[i] * 0.5f + 0.5f, 0.0f), 1.0f);
        out[i] = uint8_t(std::nearbyint(v * 255.0f));
    }
    return out;
}

std::vector<uint8_t> encode_png(std::span<const uint8_t> rgba8, uint32_t width, uint32_t height, bool rgb, bool jpeg)
{
    rgb = rgb || jpeg;
    if (rgba8.size() != size_t(width) * height * 4)
        throw std::runtime_error("encode_png: not width x height RGBA");
    std::vector<uint8_t> px(rgba8.begin(), rgba8.end());
    if (rgb)
        for (size_t i = 3; i < px.size(); i += 4)
            px[i] = 255;
    Cf<CGColorSpaceRef> cs(CGColorSpaceCreateWithName(kCGColorSpaceSRGB));
    Cf<CFDataRef> data(CFDataCreate(nullptr, px.data(), CFIndex(px.size())));
    Cf<CGDataProviderRef> prov(CGDataProviderCreateWithCFData(data.ref));
    const CGBitmapInfo info = rgb ? CGBitmapInfo(kCGImageAlphaNoneSkipLast) : CGBitmapInfo(kCGImageAlphaLast);
    Cf<CGImageRef> img(CGImageCreate(width, height, 8, 32, size_t(width) * 4, cs.ref, info, prov.ref, nullptr, false,
                                     kCGRenderingIntentDefault));
    Cf<CFMutableDataRef> out(CFDataCreateMutable(nullptr, 0));
    Cf<CGImageDestinationRef> dst(
        CGImageDestinationCreateWithData(out.ref, jpeg ? CFSTR("public.jpeg") : CFSTR("public.png"), 1, nullptr));
    if (!img.ref || !dst.ref)
        throw std::runtime_error("encode_png: ImageIO could not set up the image");
    const float quality = 0.92f;
    Cf<CFNumberRef> q(CFNumberCreate(nullptr, kCFNumberFloatType, &quality));
    const void* keys[] = {kCGImageDestinationLossyCompressionQuality};
    const void* values[] = {q.ref};
    Cf<CFDictionaryRef> props(CFDictionaryCreate(nullptr, keys, values, 1, &kCFTypeDictionaryKeyCallBacks,
                                                 &kCFTypeDictionaryValueCallBacks));
    CGImageDestinationAddImage(dst.ref, img.ref, jpeg ? props.ref : nullptr);
    if (!CGImageDestinationFinalize(dst.ref))
        throw std::runtime_error("encode_png: ImageIO could not write the PNG");
    const auto* p = CFDataGetBytePtr(out.ref);
    return std::vector<uint8_t>(p, p + CFDataGetLength(out.ref));
}

namespace {

Cf<CGImageRef> oriented(std::span<const uint8_t> data)
{
    Cf<CFDataRef> d(CFDataCreate(nullptr, data.data(), CFIndex(data.size())));
    Cf<CGImageSourceRef> src(CGImageSourceCreateWithData(d.ref, nullptr));
    if (!src.ref || CGImageSourceGetCount(src.ref) < 1)
        throw std::runtime_error("not an image ImageIO reads");
    Cf<CFDictionaryRef> props(CGImageSourceCopyPropertiesAtIndex(src.ref, 0, nullptr));
    int64_t w = 0, h = 0;
    if (props.ref) {
        if (auto n = CFNumberRef(CFDictionaryGetValue(props.ref, kCGImagePropertyPixelWidth)))
            CFNumberGetValue(n, kCFNumberSInt64Type, &w);
        if (auto n = CFNumberRef(CFDictionaryGetValue(props.ref, kCGImagePropertyPixelHeight)))
            CFNumberGetValue(n, kCFNumberSInt64Type, &h);
    }
    if (w <= 0 || h <= 0 || w > 16384 || h > 16384)
        throw std::runtime_error(std::format("an image of {}x{} pixels (at most 16384 a side)", w, h));
    // A thumbnail at full size applies the orientation the file records.
    const int64_t side = std::max(w, h);
    Cf<CFNumberRef> max(CFNumberCreate(nullptr, kCFNumberSInt64Type, &side));
    const void* keys[] = {kCGImageSourceCreateThumbnailFromImageAlways, kCGImageSourceCreateThumbnailWithTransform,
                          kCGImageSourceThumbnailMaxPixelSize};
    const void* values[] = {kCFBooleanTrue, kCFBooleanTrue, max.ref};
    Cf<CFDictionaryRef> opts(CFDictionaryCreate(nullptr, keys, values, 3, &kCFTypeDictionaryKeyCallBacks,
                                                &kCFTypeDictionaryValueCallBacks));
    Cf<CGImageRef> img(CGImageSourceCreateThumbnailAtIndex(src.ref, 0, opts.ref));
    if (!img.ref)
        throw std::runtime_error("ImageIO could not decode the image");
    return img;
}

} // namespace

std::pair<uint32_t, uint32_t> image_size(std::span<const uint8_t> data)
{
    const auto img = oriented(data);
    return {uint32_t(CGImageGetWidth(img.ref)), uint32_t(CGImageGetHeight(img.ref))};
}

std::vector<float> decode_image(std::span<const uint8_t> data, uint32_t width, uint32_t height)
{
    const auto img = oriented(data);
    const double iw = double(CGImageGetWidth(img.ref)), ih = double(CGImageGetHeight(img.ref));
    std::vector<uint8_t> px(size_t(width) * height * 4);
    Cf<CGColorSpaceRef> cs(CGColorSpaceCreateWithName(kCGColorSpaceSRGB));
    Cf<CGContextRef> ctx(CGBitmapContextCreate(px.data(), width, height, 8, size_t(width) * 4, cs.ref,
                                               CGBitmapInfo(kCGImageAlphaPremultipliedLast)));
    if (!ctx.ref)
        throw std::runtime_error("decode_image: no bitmap context");
    CGContextSetRGBFillColor(ctx.ref, 1, 1, 1, 1);
    CGContextFillRect(ctx.ref, CGRectMake(0, 0, width, height));
    CGContextSetInterpolationQuality(ctx.ref, kCGInterpolationHigh);
    const double s = std::max(width / iw, height / ih);
    CGContextDrawImage(ctx.ref, CGRectMake((width - iw * s) / 2, (height - ih * s) / 2, iw * s, ih * s), img.ref);
    std::vector<float> out(px.size());
    for (size_t i = 0; i < px.size(); ++i)
        out[i] = i % 4 == 3 ? 1.0f : px[i] / 127.5f - 1.0f;
    return out;
}

std::pair<uint32_t, uint32_t> start_size(uint32_t w, uint32_t h)
{
    const double s = std::min(1.0, std::sqrt(1024.0 * 1024.0 / (double(w) * h)));
    auto side = [&](uint32_t v) { return std::max(256u, uint32_t(std::lround(v * s / 32.0)) * 32); };
    return {side(w), side(h)};
}

void write_file(const std::string& path, std::span<const uint8_t> data)
{
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(data.data()), std::streamsize(data.size()));
    if (!f)
        throw std::runtime_error("cannot write " + path);
}

} // namespace mako::image
