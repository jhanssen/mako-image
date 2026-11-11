#include "server/image_server.hpp"

#include "http/access.hpp"
#include "image/png.hpp"
#include "log/log.hpp"

#include <glaze/glaze.hpp>
#include <httplib.h>

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <format>
#include <map>
#include <mutex>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <tuple>

namespace mako::server {

namespace {

constexpr auto kProgressEvery = std::chrono::seconds(5);
// The most reference images a request gives, and the largest one.
constexpr size_t kMaxReferences = 16;
constexpr double kMaxReferencePixels = 4096.0 * 4096;
// The largest image upscaling makes (its pixels), and the largest it takes.
constexpr double kMaxUpscaled = 4096.0 * 4096, kMaxUpscaleInput = 2048.0 * 2048;

std::string json_quoted(std::string_view s)
{
    std::string out;
    (void)glz::write_json(s, out);
    return out;
}

std::string error_body(const std::string& message, const char* type)
{
    return std::format(R"({{"error":{{"message":{},"type":{}}}}})", json_quoted(message), json_quoted(type));
}

std::string base64(std::span<const uint8_t> in)
{
    static constexpr char kTable[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((in.size() + 2) / 3 * 4);
    for (size_t i = 0; i < in.size(); i += 3) {
        const uint32_t v = uint32_t(in[i]) << 16 | (i + 1 < in.size() ? uint32_t(in[i + 1]) << 8 : 0) |
                           (i + 2 < in.size() ? uint32_t(in[i + 2]) : 0);
        out += kTable[v >> 18];
        out += kTable[(v >> 12) & 63];
        out += i + 1 < in.size() ? kTable[(v >> 6) & 63] : '=';
        out += i + 2 < in.size() ? kTable[v & 63] : '=';
    }
    return out;
}

// Base64 (a "data:...;base64," prefix allowed); nullopt on anything else.
std::optional<std::vector<uint8_t>> unbase64(std::string_view s)
{
    if (s.starts_with("data:")) {
        const auto comma = s.find(',');
        if (comma == s.npos || s.substr(0, comma).find(";base64") == s.npos)
            return std::nullopt;
        s.remove_prefix(comma + 1);
    }
    std::vector<uint8_t> out;
    out.reserve(s.size() / 4 * 3);
    uint32_t acc = 0, bits = 0;
    for (const char ch : s) {
        int v = ch >= 'A' && ch <= 'Z' ? ch - 'A' : ch >= 'a' && ch <= 'z' ? ch - 'a' + 26
              : ch >= '0' && ch <= '9' ? ch - '0' + 52 : ch == '+' || ch == '-' ? 62 : ch == '/' || ch == '_' ? 63 : -1;
        if (ch == '=')
            break;
        if (v < 0) {
            if (ch == '\n' || ch == '\r' || ch == ' ')
                continue;
            return std::nullopt;
        }
        acc = acc << 6 | uint32_t(v);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(uint8_t(acc >> bits));
        }
    }
    return out;
}

} // namespace

struct ImageServer::Impl {
    Impl(ImageResidency& r, LoadImage l, ImageServerOptions o, LoadUpscaler u)
        : residency(r), load(std::move(l)), opts(std::move(o)), load_upscaler(std::move(u))
    {
    }
    ImageResidency& residency;
    LoadImage load;
    ImageServerOptions opts;
    LoadUpscaler load_upscaler;
    httplib::Server http;
    std::atomic<uint32_t> in_flight = 0, next_id = 0;
    std::atomic<int> port = 0;
    // The upscalers loaded so far, by name (small: they stay), and one upscale at a time.
    std::mutex upscale_mutex;
    std::map<std::string, std::unique_ptr<image::Esrgan>> upscalers;

    void images(const httplib::Request& req, httplib::Response& res);
    void loras(const httplib::Request& req, httplib::Response& res);
    void image_model_list(httplib::Response& res);
    ImageModel image_model(const std::string& name) const;

    // What a request asks to upscale with: the upscaler and the size multiple (0: the upscaler's own).
    struct UpscaleAsk {
        UpscalerChoice model;
        float scale = 0;
    };
    // v: an upscaler's name, or an object with "model" (default: the first by name) and "scale" (strict: no other
    // fields). Throws std::invalid_argument naming the problem.
    UpscaleAsk upscale_ask(const glz::generic& v, bool strict) const;
    // Why an image of w x h at the ask's scale cannot be made ("" when it can).
    static std::string too_big(const UpscaleAsk& ask, uint32_t w, uint32_t h);
    // RGBA [h][w][4] in [-1, 1] upscaled as asked, the upscaler loaded if need be: the image, its size and the scale in
    // effect.
    struct Upscaled {
        std::vector<float> rgba;
        uint32_t width = 0, height = 0;
        float scale = 1;
    };
    Upscaled upscale(const UpscaleAsk& ask, std::span<const float> rgba, uint32_t w, uint32_t h);
    void upscales(const httplib::Request& req, httplib::Response& res);
    void upscaler_list(httplib::Response& res);
};

ImageServer::Impl::UpscaleAsk ImageServer::Impl::upscale_ask(const glz::generic& v, bool strict) const
{
    UpscaleAsk ask;
    std::string name;
    if (v.is_string()) {
        name = v.get_string();
    } else if (v.is_object()) {
        if (strict)
            for (const auto& [k, x] : v.get_object())
                if (k != "model" && k != "scale")
                    throw std::invalid_argument(std::format("unknown field {}", k));
        if (v.contains("model") && !v["model"].is_null()) {
            if (!v["model"].is_string())
                throw std::invalid_argument("model must be a string");
            name = v["model"].get_string();
        }
        if (v.contains("scale") && !v["scale"].is_null()) {
            if (!v["scale"].is_number() || !(v["scale"].get_number() >= 1) || v["scale"].get_number() > 8)
                throw std::invalid_argument("scale must be a number from 1 to 8");
            ask.scale = float(v["scale"].get_number());
        }
    } else {
        throw std::invalid_argument("an upscaler's name, or {\"model\": NAME, \"scale\": F}");
    }
    if (opts.upscalers.empty())
        throw std::invalid_argument("this server has no upscalers (mako_image_serve --upscalers DIR)");
    if (name.ends_with(".safetensors"))
        name.resize(name.size() - std::string_view(".safetensors").size());
    const auto list = upscaler_choices(opts.upscalers);
    if (name.empty()) {
        if (list.empty())
            throw std::invalid_argument(std::format("no upscalers in {}", opts.upscalers));
        ask.model = list.front();
    } else {
        const auto it = std::ranges::find(list, name, &UpscalerChoice::name);
        if (it == list.end())
            throw std::invalid_argument(
                std::format("no upscaler named \"{}\" (GET /v1/images/upscalers lists them)", name));
        ask.model = *it;
    }
    if (ask.scale > float(ask.model.factor))
        throw std::invalid_argument(std::format("scale: {} goes up to {}x", ask.model.name, ask.model.factor));
    return ask;
}

std::string ImageServer::Impl::too_big(const UpscaleAsk& ask, uint32_t w, uint32_t h)
{
    const double s = ask.scale ? ask.scale : ask.model.factor;
    if (double(w) * h * s * s > kMaxUpscaled)
        return std::format("{}x{} at {:g}x is past 4096x4096 pixels' area (a smaller scale)", w, h, s);
    return {};
}

ImageServer::Impl::Upscaled ImageServer::Impl::upscale(const UpscaleAsk& ask, std::span<const float> rgba, uint32_t w,
                                                       uint32_t h)
{
    std::lock_guard lock(upscale_mutex);
    auto it = upscalers.find(ask.model.name);
    if (it == upscalers.end() || it->second->path() != ask.model.path) {
        if (!load_upscaler)
            throw std::runtime_error("this server cannot load upscalers");
        const auto t0 = std::chrono::steady_clock::now();
        it = upscalers.insert_or_assign(ask.model.name, load_upscaler(ask.model.path)).first;
        log::info("upscaler {} loaded in {:.1f} s", ask.model.name,
                  std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    image::Esrgan& up = *it->second;
    const uint32_t f = up.factor();
    Upscaled o;
    o.scale = ask.scale ? ask.scale : float(f);
    o.width = uint32_t(std::lround(w * o.scale));
    o.height = uint32_t(std::lround(h * o.scale));
    o.rgba = up.upscale(rgba, w, h);
    if (o.width != w * f || o.height != h * f)
        o.rgba = image::resize_image(o.rgba, w * f, h * f, o.width, o.height);
    return o;
}

void ImageServer::Impl::upscales(const httplib::Request& req, httplib::Response& res)
{
    auto fail = [&](int status, const std::string& message, const char* type) {
        res.status = status;
        res.set_content(error_body(message, type), "application/json");
    };
    glz::generic body;
    if (glz::read_json(body, req.body) || !body.is_object())
        return fail(400, "the body is not a JSON object", "invalid_request_error");
    for (const auto& [k, v] : body.get_object())
        if (k != "image" && k != "model" && k != "scale" && k != "output_format" && k != "response_format")
            return fail(400, std::format("unknown field {}", k), "invalid_request_error");
    if (!body.contains("image") || body["image"].is_null())
        return fail(400, "image is required", "invalid_request_error");
    const auto bytes = body["image"].is_string() ? unbase64(body["image"].get_string()) : std::nullopt;
    if (!bytes || bytes->empty())
        return fail(400, "image must be an image file in base64", "invalid_request_error");
    if (body.contains("response_format") && body["response_format"].is_string() &&
        body["response_format"].get_string() != "b64_json")
        return fail(400, "response_format: only b64_json (images are not stored)", "invalid_request_error");
    bool jpeg = false;
    if (body.contains("output_format") && body["output_format"].is_string()) {
        const std::string f = body["output_format"].get_string();
        if (f != "png" && f != "jpeg")
            return fail(400, "output_format: png or jpeg", "invalid_request_error");
        jpeg = f == "jpeg";
    }
    UpscaleAsk ask;
    try {
        ask = upscale_ask(body, false);
    } catch (const std::exception& e) {
        return fail(400, e.what(), "invalid_request_error");
    }
    uint32_t width = 0, height = 0;
    std::vector<float> rgba;
    try {
        std::tie(width, height) = image::image_size(*bytes);
        if (double(width) * height > kMaxUpscaleInput)
            return fail(400, std::format("image: {}x{} is past 2048x2048 pixels' area", width, height),
                        "invalid_request_error");
        rgba = image::decode_image(*bytes, width, height);
    } catch (const std::exception& e) {
        return fail(400, std::format("image: {}", e.what()), "invalid_request_error");
    }
    if (const std::string why = too_big(ask, width, height); !why.empty())
        return fail(400, why, "invalid_request_error");
    if (in_flight.fetch_add(1) > opts.max_waiting) {
        --in_flight;
        return fail(503, std::format("{} requests are already waiting", opts.max_waiting), "server_busy");
    }
    const auto slot = std::shared_ptr<void>(nullptr, [this](void*) { --in_flight; });
    const std::string id = std::format("up-{}", next_id.fetch_add(1) + 1);
    log::info("{}: upscale {}x{} with {}{}", id, width, height, ask.model.name,
              ask.scale ? std::format(" to {:g}x", ask.scale) : "");
    std::string data;
    Upscaled u;
    try {
        const auto t0 = std::chrono::steady_clock::now();
        u = upscale(ask, rgba, width, height);
        const auto png = image::encode_png(image::to_rgba8(u.rgba), u.width, u.height, false, jpeg);
        log::info("{}: {}x{} in {:.1f} s", id, u.width, u.height,
                  std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        data = std::format(R"({{"b64_json":"{}"}})", base64(png));
    } catch (const std::exception& e) {
        log::error("{}: {}", id, e.what());
        return fail(500, e.what(), "server_error");
    }
    const int64_t created =
        std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    res.set_content(std::format(R"({{"created":{},"data":[{}],"output_format":"{}","size":"{}x{}"}})", created, data,
                                jpeg ? "jpeg" : "png", u.width, u.height),
                    "application/json");
}

void ImageServer::Impl::upscaler_list(httplib::Response& res)
{
    std::string data;
    for (const auto& c : upscaler_choices(opts.upscalers))
        data += std::format(R"({}{{"name":{},"factor":{},"blocks":{}{}}})", data.empty() ? "" : ",",
                            json_quoted(c.name), c.factor, c.blocks,
                            c.description.empty() ? "" : std::format(R"(,"description":{})", json_quoted(c.description)));
    res.set_content(std::format(R"({{"object":"list","data":[{}]}})", data), "application/json");
}

// Image prompts' text encoders and caches grow with the prompt.
constexpr size_t kMaxImagePrompt = 16384;

void ImageServer::Impl::images(const httplib::Request& req, httplib::Response& res)
{
    auto fail = [&](int status, const std::string& message, const char* type) {
        res.status = status;
        res.set_content(error_body(message, type), "application/json");
    };
    glz::generic body;
    if (glz::read_json(body, req.body) || !body.is_object())
        return fail(400, "the body is not a JSON object", "invalid_request_error");
    if (!body.contains("prompt") || !body["prompt"].is_string() || body["prompt"].get_string().empty())
        return fail(400, "prompt is required", "invalid_request_error");
    const std::string prompt = body["prompt"].get_string();
    if (prompt.size() > kMaxImagePrompt)
        return fail(400, std::format("prompt is longer than {} bytes", kMaxImagePrompt), "invalid_request_error");
    uint32_t n = 1, width = 1024, height = 1024;
    bool sized = false;
    if (body.contains("n")) {
        if (!body["n"].is_number() || body["n"].get_number() < 1 || body["n"].get_number() > 4)
            return fail(400, "n must be 1 to 4", "invalid_request_error");
        n = uint32_t(body["n"].get_number());
    }
    if (body.contains("size") && body["size"].is_string() && body["size"].get_string() != "auto") {
        const std::string sz = body["size"].get_string();
        unsigned w = 0, h = 0;
        char x = 0;
        if (std::sscanf(sz.c_str(), "%u%c%u", &w, &x, &h) != 3 || x != 'x' || w % 32 || h % 32 || w < 256 ||
            h < 256)
            return fail(400, "size must be WIDTHxHEIGHT, multiples of 32 from 256", "invalid_request_error");
        width = w;
        height = h;
        sized = true;
    }
    if (body.contains("response_format") && body["response_format"].is_string() &&
        body["response_format"].get_string() != "b64_json")
        return fail(400, "response_format: only b64_json (images are not stored)", "invalid_request_error");
    // Upscaled afterwards: an upscaler's name, or {"model": NAME, "scale": F} (the size checked once it is known).
    std::optional<UpscaleAsk> up;
    if (body.contains("upscale") && !body["upscale"].is_null()) {
        try {
            up = upscale_ask(body["upscale"], true);
        } catch (const std::exception& e) {
            return fail(400, std::format("upscale: {}", e.what()), "invalid_request_error");
        }
    }
    // Extensions: a negative prompt (classifier-free guidance) and its scale.
    std::optional<std::string> negative;
    ImageModel model;
    try {
        model = image_model(body.contains("model") && body["model"].is_string() ? body["model"].get_string() : "");
    } catch (const std::exception& e) {
        return fail(400, e.what(), "invalid_request_error");
    }
    if (uint64_t(width) * height > image::Engine::max_pixels(model.family))
        return fail(400, std::format("size: {} makes at most {} megapixels", model.name,
                                     double(image::Engine::max_pixels(model.family)) / 1e6),
                    "invalid_request_error");
    const bool fixed = image::Engine::fixed_schedule(model.family);
    std::optional<float> guidance = model.guidance;
    if (body.contains("negative_prompt") && !body["negative_prompt"].is_null()) {
        if (!body["negative_prompt"].is_string())
            return fail(400, "negative_prompt must be a string", "invalid_request_error");
        negative = body["negative_prompt"].get_string();
        if (negative->size() > kMaxImagePrompt)
            return fail(400, std::format("negative_prompt is longer than {} bytes", kMaxImagePrompt),
                        "invalid_request_error");
    }
    if (body.contains("guidance_scale") && !body["guidance_scale"].is_null()) {
        if (!body["guidance_scale"].is_number() || body["guidance_scale"].get_number() < 0 ||
            body["guidance_scale"].get_number() > 30)
            return fail(400, "guidance_scale must be a number from 0 to 30", "invalid_request_error");
        guidance = float(body["guidance_scale"].get_number());
    }
    std::optional<image::Sampler> sampler = model.sampler;
    const bool asked_sampler = body.contains("sampler") && !body["sampler"].is_null();
    if (asked_sampler) {
        sampler = body["sampler"].is_string() ? image::parse_sampler(body["sampler"].get_string()) : std::nullopt;
        if (!sampler)
            return fail(400, "sampler must be euler, euler_a or er_sde", "invalid_request_error");
        if (sampler == image::Sampler::euler_ancestral && !image::Engine::predicts_noise(model.family))
            return fail(400, std::format("sampler: euler_a needs a model that predicts the noise ({} does not)",
                                         model.name),
                        "invalid_request_error");
    }
    if (!model.options.adapter.empty() && ((guidance && *guidance != 1.0f) || sampler))
        return fail(400, std::format("{} runs its adapter's sampler, without guidance: no guidance_scale or sampler",
                                     model.name),
                    "invalid_request_error");
    std::optional<std::vector<uint32_t>> skip_blocks = model.skip_blocks;
    if (body.contains("skip_blocks") && !body["skip_blocks"].is_null()) {
        std::string text;
        (void)glz::write_json(body["skip_blocks"], text);
        skip_blocks = parse_skip_blocks(text);
        if (!skip_blocks)
            return fail(400, "skip_blocks must be an array of block numbers", "invalid_request_error");
        if (!skip_blocks->empty() && !image::Engine::skips_blocks(model.family))
            return fail(400, std::format("skip_blocks: {} has no blocks to skip", model.name), "invalid_request_error");
    }
    uint32_t steps = model.steps;
    if (body.contains("steps") && !body["steps"].is_null()) {
        const auto& v = body["steps"];
        if (!v.is_number() || v.get_number() < 0 || v.get_number() > 100 || v.get_number() != std::floor(v.get_number()))
            return fail(400, "steps must be a whole number from 0 (the model's) to 100", "invalid_request_error");
        steps = uint32_t(v.get_number());
    }
    if (fixed && (negative || (guidance && *guidance != 1.0f) || (sampler && *sampler != image::Sampler::euler)))
        return fail(400, std::format("{} samples with Euler on its own schedule, without guidance: no negative_prompt, "
                                     "guidance_scale or sampler", model.name),
                    "invalid_request_error");
    // The smallest canvas a model that generates larger than asked uses.
    std::optional<uint32_t> min_area = model.min_area;
    if (body.contains("min_generation_area") && !body["min_generation_area"].is_null()) {
        const auto& v = body["min_generation_area"];
        if (!image::Engine::generates_on_canvas(model.family))
            return fail(400, std::format("min_generation_area: {} generates at the size asked for", model.name),
                        "invalid_request_error");
        if (!v.is_number() || v.get_number() < 0 || v.get_number() > 4096.0 * 4096 || v.get_number() != std::floor(v.get_number()))
            return fail(400, "min_generation_area must be a whole number of pixels up to 4096x4096 (0: the size asked for)",
                        "invalid_request_error");
        min_area = uint32_t(v.get_number());
    }
    // Reference images (models that edit with them): image files in base64, <Picture 1>, <Picture 2>, ... in order.
    std::vector<std::vector<float>> ref_rgba;
    std::vector<image::Engine::Request::Reference> references;
    if (body.contains("references") && !body["references"].is_null()) {
        const auto& v = body["references"];
        if (!v.is_array())
            return fail(400, "references must be an array of image files in base64", "invalid_request_error");
        if (!image::Engine::takes_references(model.family))
            return fail(400, std::format("references: {} takes no reference images", model.name), "invalid_request_error");
        if (v.get_array().size() > kMaxReferences)
            return fail(400, std::format("references: at most {}", kMaxReferences), "invalid_request_error");
        for (const auto& e : v.get_array()) {
            const auto bytes = e.is_string() ? unbase64(e.get_string()) : std::nullopt;
            if (!bytes || bytes->empty())
                return fail(400, std::format("references[{}] must be an image file in base64", references.size()),
                            "invalid_request_error");
            try {
                const auto [w, h] = image::image_size(*bytes);
                if (double(w) * h > kMaxReferencePixels || w < 32 || h < 32)
                    return fail(400, std::format("references[{}]: {}x{} is outside 32 pixels a side to 16 megapixels",
                                                 references.size(), w, h),
                                "invalid_request_error");
                ref_rgba.push_back(image::decode_image(*bytes, w, h));
                references.push_back({w, h, ref_rgba.back()});
            } catch (const std::exception& ex) {
                return fail(400, std::format("references[{}]: {}", references.size(), ex.what()), "invalid_request_error");
            }
        }
    }
    image::Loras loras = model.loras;
    if (body.contains("loras") && !body["loras"].is_null()) {
        try {
            std::string text;
            (void)glz::write_json(body["loras"], text);
            loras = resolve_loras(text, lora_choices(model.loras, model.lora_dir));
        } catch (const std::exception& e) {
            return fail(400, std::format("loras: {}", e.what()), "invalid_request_error");
        }
        for (const auto& l : loras)
            if (l.merge == false && !image::Engine::keeps_loras_unmerged(model.family))
                return fail(400, std::format("loras: {} keeps no LoRA unmerged", model.name), "invalid_request_error");
    }
    // A 64-bit seed (as images record it) is past a JSON number's exact range, so it may come as a string.
    std::optional<uint64_t> seed;
    if (body.contains("seed") && !body["seed"].is_null()) {
        const auto& v = body["seed"];
        uint64_t x = 0;
        if (v.is_number() && v.get_number() >= 0 && v.get_number() <= 9007199254740992.0 &&
            v.get_number() == std::floor(v.get_number()))
            x = uint64_t(v.get_number());
        else if (const std::string t = v.is_string() ? v.get_string() : std::string();
                 t.empty() || std::from_chars(t.data(), t.data() + t.size(), x).ptr != t.data() + t.size())
            return fail(400, "seed must be a whole number from 0 (a string past 2^53)", "invalid_request_error");
        seed = x;
    }
    // Image to image: the image to start from (base64) and the share of the steps run on it.
    std::vector<float> start;
    float strength = 0.75f;
    if (body.contains("image") && !body["image"].is_null()) {
        const auto bytes = body["image"].is_string() ? unbase64(body["image"].get_string()) : std::nullopt;
        if (!bytes || bytes->empty())
            return fail(400, "image must be an image file in base64", "invalid_request_error");
        if (!model.options.adapter.empty())
            return fail(400, std::format("image: {} runs its adapter's sampler, from noise only", model.name),
                        "invalid_request_error");
        if (!image::Engine::encodes_images(model.family))
            return fail(400, std::format("image: {} cannot start from an image yet", model.name),
                        "invalid_request_error");
        if (body.contains("strength") && !body["strength"].is_null()) {
            const auto& v = body["strength"];
            if (!v.is_number() || !(v.get_number() > 0) || v.get_number() > 1)
                return fail(400, "strength must be a number above 0, at most 1", "invalid_request_error");
            strength = float(v.get_number());
        }
        try {
            if (!sized) {
                const auto [w, h] = image::image_size(*bytes);
                std::tie(width, height) = image::start_size(w, h);
            }
            start = image::decode_image(*bytes, width, height);
        } catch (const std::exception& e) {
            return fail(400, std::format("image: {}", e.what()), "invalid_request_error");
        }
    } else if (body.contains("strength") && !body["strength"].is_null()) {
        return fail(400, "strength goes with an image", "invalid_request_error");
    }
    // Control maps (a model with a control adapter): {"color": BASE64, "depth": BASE64, "color_end": F, "depth_end": F}.
    std::vector<float> color_map, depth_map;
    float color_end = 1.0f, depth_end = 1.0f;
    if (body.contains("control") && !body["control"].is_null()) {
        const auto& c = body["control"];
        if (!c.is_object())
            return fail(400, "control must be an object", "invalid_request_error");
        if (model.options.control.empty())
            return fail(400, std::format("control: {} has no control adapter", model.name), "invalid_request_error");
        for (const auto& [k, v] : c.get_object())
            if (k != "color" && k != "depth" && k != "color_end" && k != "depth_end")
                return fail(400, std::format("control: unknown field {}", k), "invalid_request_error");
        for (auto [key, end] : {std::pair{"color_end", &color_end}, {"depth_end", &depth_end}})
            if (c.contains(key) && !c[key].is_null()) {
                if (!c[key].is_number() || c[key].get_number() < 0 || c[key].get_number() > 1)
                    return fail(400, std::format("control.{} must be a number from 0 to 1", key),
                                "invalid_request_error");
                *end = float(c[key].get_number());
            }
        for (auto [key, out] : {std::pair{"color", &color_map}, {"depth", &depth_map}}) {
            if (!c.contains(key) || c[key].is_null())
                continue;
            const auto bytes = c[key].is_string() ? unbase64(c[key].get_string()) : std::nullopt;
            if (!bytes || bytes->empty())
                return fail(400, std::format("control.{} must be an image file in base64", key),
                            "invalid_request_error");
            try {
                if (!sized && start.empty()) { // the first map's size, as an image to start from gives its own
                    const auto [w, h] = image::image_size(*bytes);
                    std::tie(width, height) = image::start_size(w, h);
                    sized = true;
                }
                *out = image::decode_image(*bytes, width, height);
            } catch (const std::exception& e) {
                return fail(400, std::format("control.{}: {}", key, e.what()), "invalid_request_error");
            }
        }
    }
    bool jpeg = false;
    if (body.contains("output_format") && body["output_format"].is_string()) {
        const std::string f = body["output_format"].get_string();
        if (f != "png" && f != "jpeg")
            return fail(400, "output_format: png or jpeg", "invalid_request_error");
        jpeg = f == "jpeg";
    }
    if (up)
        if (const std::string why = too_big(*up, width, height); !why.empty())
            return fail(400, "upscale: " + why, "invalid_request_error");
    if (in_flight.fetch_add(1) > opts.max_waiting) {
        --in_flight;
        return fail(503, std::format("{} requests are already waiting", opts.max_waiting), "server_busy");
    }
    const auto slot = std::shared_ptr<void>(nullptr, [this](void*) { --in_flight; });
    const std::string id = std::format("img-{}", next_id.fetch_add(1) + 1);
    std::string lora_names;
    for (const auto& l : loras)
        lora_names += std::format("{}{} {:g}{}", lora_names.empty() ? ", LoRAs " : ", ",
                                  std::filesystem::path(l.path).stem().string(), l.scale,
                                  !l.merge ? "" : *l.merge ? " merged" : " unmerged");
    log::info("{}: {} image{} {}x{} with {}{}{}{}{}{}, waiting for the image model", id, n, n > 1 ? "s" : "", width,
              height, model.name, negative ? ", a negative prompt" : "", lora_names,
              asked_sampler ? std::format(", sampler {}", image::sampler_name(*sampler)) : "",
              start.empty() ? "" : std::format(", from an image (strength {:g})", strength),
              references.empty() ? "" : std::format(", {} reference image{}", references.size(), references.size() > 1 ? "s" : ""));
    std::string data = "[";
    uint32_t out_w = width, out_h = height; // the images' size (upscaled, if asked)
    try {
        std::string key = model.key();
        for (const auto& l : loras)
            key += std::format("|{}:{}:{}", l.path, l.scale, !l.merge ? "" : *l.merge ? "merge" : "unmerged");
        auto engine = residency.acquire(key, "image model " + model.name, [&] { return load(model, loras); });
        std::random_device rd;
        for (uint32_t i = 0; i < n; ++i) {
            image::Engine::Request r;
            r.prompt = prompt;
            r.width = width;
            r.height = height;
            r.seed = seed ? *seed + i : (uint64_t(rd()) << 32) | rd();
            r.steps = steps;
            r.terminal = model.terminal;
            r.negative = negative;
            r.guidance = guidance;
            r.sampler = sampler;
            r.skip_blocks = skip_blocks;
            r.image = start;
            r.strength = strength;
            r.control = {color_map, depth_map, color_end, depth_end};
            r.references = references;
            r.min_area = min_area;
            auto last = std::chrono::steady_clock::now();
            const auto t0 = last;
            r.progress = [&](uint32_t step, uint32_t steps) {
                const auto now = std::chrono::steady_clock::now();
                if (now - last < kProgressEvery && step < steps)
                    return;
                last = now;
                const double s = std::chrono::duration<double>(now - t0).count();
                log::info("{}: image {} of {}, step {} of {} ({:.1f} s a step, ~{:.0f} s left)", id, i + 1, n, step,
                          steps, s / step, s / step * (steps - step));
            };
            auto img = engine->generate(r);
            image::Provenance prov = engine->provenance(r);
            if (up) {
                const auto t_up = std::chrono::steady_clock::now();
                Upscaled u = upscale(*up, img, width, height);
                img = std::move(u.rgba);
                out_w = u.width;
                out_h = u.height;
                prov.upscale = {up->model.name, u.scale};
                log::info("{}: image {} of {} upscaled to {}x{} with {} in {:.1f} s", id, i + 1, n, out_w, out_h,
                          up->model.name, std::chrono::duration<double>(std::chrono::steady_clock::now() - t_up).count());
            }
            auto bytes = image::encode_png(image::to_rgba8(img), out_w, out_h, false, jpeg);
            if (!jpeg) {
                prov.model = model.name;
                for (const auto& l : loras)
                    prov.loras.push_back({std::filesystem::path(l.path).stem().string(), l.scale, l.merge});
                bytes = image::with_provenance(std::move(bytes), prov);
            }
            log::info("{}: image {} of {} in {:.1f} s (seed {})", id, i + 1, n,
                      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), r.seed);
            data += std::format(R"({}{{"b64_json":"{}"}})", i ? "," : "", base64(bytes));
        }
    } catch (const std::exception& e) {
        log::error("{}: {}", id, e.what());
        return fail(500, e.what(), "server_error");
    }
    data += "]";
    const int64_t created =
        std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    res.set_content(std::format(R"({{"created":{},"data":{},"output_format":"{}","size":"{}x{}"}})", created, data,
                                jpeg ? "jpeg" : "png", out_w, out_h),
                    "application/json");
}

ImageModel ImageServer::Impl::image_model(const std::string& name) const
{
    if (!opts.image_models.empty() && (!name.empty() || !std::filesystem::is_directory(opts.image_model))) {
        const std::string want = name.empty() ? opts.image_model : name;
        auto list = image_models(opts.image_models);
        if (want.empty() && !list.empty()) // no default named: the first
            return list.front();
        for (auto& m : list)
            if (m.name == want)
                return m;
        throw std::invalid_argument(std::format("no image model named \"{}\" (GET /v1/images/models lists them)", want));
    }
    const std::string dir = opts.image_model;
    const std::string own = std::filesystem::path(dir).lexically_normal().filename().string();
    if (!name.empty() && name != own)
        throw std::invalid_argument(std::format("no image model named \"{}\" (this server has only {})", name, own));
    return load_image_model(dir, own);
}

void ImageServer::Impl::image_model_list(httplib::Response& res)
{
    std::vector<ImageModel> list;
    if (!opts.image_models.empty())
        list = image_models(opts.image_models);
    std::string def, loaded = residency.loaded();
    try {
        const ImageModel d = image_model("");
        def = d.name;
        if (!std::ranges::contains(list, d.name, &ImageModel::name))
            list.insert(list.begin(), d);
    } catch (const std::exception& e) {
        log::warn("the default image model: {}", e.what());
    }
    std::string data;
    for (const ImageModel& m : list) {
        data += std::format(R"({}{{"id":{},"object":"model","default":{},"loaded":{})", data.empty() ? "" : ",",
                            json_quoted(m.name), m.name == def, m.name == loaded);
        data += std::format(R"(,"family":{})", json_quoted(m.family));
        if (!m.description.empty())
            data += std::format(R"(,"description":{})", json_quoted(m.description));
        if (!m.prompting.empty())
            data += std::format(R"(,"prompting":{})", json_quoted(m.prompting));
        if (m.steps)
            data += std::format(R"(,"steps":{})", m.steps);
        if (m.guidance)
            data += std::format(R"(,"guidance":{:g})", *m.guidance);
        if (m.sampler)
            data += std::format(R"(,"sampler":"{}")", image::sampler_name(*m.sampler));
        if (m.skip_blocks) {
            std::string b;
            for (uint32_t i : *m.skip_blocks)
                b += std::format("{}{}", b.empty() ? "" : ",", i);
            data += std::format(R"(,"skip_blocks":[{}])", b);
        }
        if (image::Engine::encodes_images(m.family) && m.options.adapter.empty())
            data += R"(,"image_to_image":true)";
        if (!m.options.control.empty())
            data += R"(,"control":true)";
        if (image::Engine::takes_references(m.family))
            data += R"(,"references":true)";
        if (image::Engine::fixed_schedule(m.family))
            data += R"(,"fixed_schedule":true)";
        if (m.min_area)
            data += std::format(R"(,"min_generation_area":{})", *m.min_area);
        std::string l;
        for (const auto& x : m.loras)
            l += std::format(R"({}{{"name":{},"scale":{:g}{}}})", l.empty() ? "" : ",",
                             json_quoted(std::filesystem::path(x.path).stem().string()), x.scale,
                             x.merge ? std::format(R"(,"merge":{})", *x.merge) : "");
        data += std::format(R"(,"loras":[{}]}})", l);
    }
    res.set_content(std::format(R"({{"object":"list","data":[{}]}})", data), "application/json");
}

void ImageServer::Impl::loras(const httplib::Request& req, httplib::Response& res)
{
    ImageModel model;
    try {
        model = image_model(req.has_param("model") ? req.get_param_value("model") : "");
    } catch (const std::exception& e) {
        res.status = 400;
        res.set_content(error_body(e.what(), "invalid_request_error"), "application/json");
        return;
    }
    std::string data;
    for (const auto& c : lora_choices(model.loras, model.lora_dir)) {
        data += std::format(R"({}{{"name":{})", data.empty() ? "" : ",", json_quoted(c.name));
        if (c.scale)
            data += std::format(R"(,"default_scale":{:g})", *c.scale);
        if (c.merge)
            data += std::format(R"(,"default_merge":{})", *c.merge);
        const auto notes = lora_notes(c.path);
        if (!notes.trigger.empty())
            data += std::format(R"(,"trigger":{})", json_quoted(notes.trigger));
        if (!notes.description.empty())
            data += std::format(R"(,"description":{})", json_quoted(notes.description));
        if (notes.scale_range)
            data += std::format(R"(,"scale_range":[{:g},{:g}])", notes.scale_range->first, notes.scale_range->second);
        data += "}";
    }
    res.set_content(std::format(R"({{"object":"list","data":[{}]}})", data), "application/json");
}


ImageServer::ImageServer(ImageResidency& residency, LoadImage load, ImageServerOptions opts, LoadUpscaler load_upscaler)
    : impl_(std::make_unique<Impl>(residency, std::move(load), std::move(opts), std::move(load_upscaler)))
{
    auto& h = impl_->http;
    h.set_pre_routing_handler([this](const httplib::Request& req, httplib::Response& res) {
        const auto& keys = impl_->opts.api_keys;
        if (req.path == "/health" ||
            http::allowed(req.get_header_value("Authorization"), req.get_header_value("x-api-key"),
                          req.get_header_value("Host"), req.get_header_value("Origin"), keys))
            return httplib::Server::HandlerResponse::Unhandled;
        res.status = keys.empty() ? 403 : 401;
        res.set_content(error_body(keys.empty() ? "without an API key, requests from web pages or other hosts are refused"
                                                : "invalid API key",
                                   keys.empty() ? "permission_denied" : "invalid_api_key"),
                        "application/json");
        return httplib::Server::HandlerResponse::Handled;
    });
    const auto images = [this](const httplib::Request& req, httplib::Response& res) { impl_->images(req, res); };
    h.Post("/v1/images/generations", images);
    h.Post("/images/generations", images);
    const auto loras = [this](const httplib::Request& req, httplib::Response& res) { impl_->loras(req, res); };
    h.Get("/v1/images/loras", loras);
    h.Get("/images/loras", loras);
    const auto list = [this](const httplib::Request&, httplib::Response& res) { impl_->image_model_list(res); };
    h.Get("/v1/images/models", list);
    h.Get("/images/models", list);
    const auto upscales = [this](const httplib::Request& req, httplib::Response& res) { impl_->upscales(req, res); };
    h.Post("/v1/images/upscales", upscales);
    h.Post("/images/upscales", upscales);
    const auto upscalers = [this](const httplib::Request&, httplib::Response& res) { impl_->upscaler_list(res); };
    h.Get("/v1/images/upscalers", upscalers);
    h.Get("/images/upscalers", upscalers);
    h.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(R"({"status":"ok"})", "application/json");
    });
    h.Get("/v1/status", [this](const httplib::Request&, httplib::Response& res) {
        auto& im = *impl_;
        res.set_content(std::format(R"({{"model":{},"state":"{}","requests":{}}})", json_quoted(im.residency.loaded()),
                                    im.residency.state(), im.in_flight.load()),
                        "application/json");
    });
}

ImageServer::~ImageServer() = default;

bool ImageServer::run()
{
    auto& im = *impl_;
    const int port = im.opts.port ? (im.http.bind_to_port(im.opts.host, im.opts.port) ? im.opts.port : -1)
                                  : im.http.bind_to_any_port(im.opts.host);
    if (port < 0)
        return false;
    im.port = port;
    return im.http.listen_after_bind();
}

void ImageServer::stop()
{
    impl_->http.stop();
}

int ImageServer::port() const
{
    return impl_->port;
}

ImageModel ImageServer::model(const std::string& name) const
{
    return impl_->image_model(name);
}

} // namespace mako::server
