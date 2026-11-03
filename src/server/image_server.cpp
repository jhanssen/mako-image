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
#include <random>
#include <span>
#include <stdexcept>

namespace mako::server {

namespace {

constexpr auto kProgressEvery = std::chrono::seconds(5);

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

} // namespace

struct ImageServer::Impl {
    Impl(ImageResidency& r, LoadImage l, ImageServerOptions o) : residency(r), load(std::move(l)), opts(std::move(o)) {}
    ImageResidency& residency;
    LoadImage load;
    ImageServerOptions opts;
    httplib::Server http;
    std::atomic<uint32_t> in_flight = 0, next_id = 0;
    std::atomic<int> port = 0;

    void images(const httplib::Request& req, httplib::Response& res);
    void loras(const httplib::Request& req, httplib::Response& res);
    void image_model_list(httplib::Response& res);
    ImageModel image_model(const std::string& name) const;
};

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
            h < 256 || uint64_t(w) * h > 2048ull * 2048)
            return fail(400, "size must be WIDTHxHEIGHT, multiples of 32 from 256, at most 2048x2048 pixels",
                        "invalid_request_error");
        width = w;
        height = h;
    }
    if (body.contains("response_format") && body["response_format"].is_string() &&
        body["response_format"].get_string() != "b64_json")
        return fail(400, "response_format: only b64_json (images are not stored)", "invalid_request_error");
    // Extensions: a negative prompt (classifier-free guidance) and its scale.
    std::optional<std::string> negative;
    ImageModel model;
    try {
        model = image_model(body.contains("model") && body["model"].is_string() ? body["model"].get_string() : "");
    } catch (const std::exception& e) {
        return fail(400, e.what(), "invalid_request_error");
    }
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
    if (body.contains("sampler") && !body["sampler"].is_null()) {
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
    bool jpeg = false;
    if (body.contains("output_format") && body["output_format"].is_string()) {
        const std::string f = body["output_format"].get_string();
        if (f != "png" && f != "jpeg")
            return fail(400, "output_format: png or jpeg", "invalid_request_error");
        jpeg = f == "jpeg";
    }
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
    log::info("{}: {} image{} {}x{} with {}{}{}, waiting for the image model", id, n, n > 1 ? "s" : "", width, height,
              model.name, negative ? ", a negative prompt" : "", lora_names);
    std::string data = "[";
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
            const auto img = engine->generate(r);
            auto bytes = image::encode_png(image::to_rgba8(img), width, height, false, jpeg);
            if (!jpeg) {
                image::Provenance prov = engine->provenance(r);
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
                                jpeg ? "jpeg" : "png", width, height),
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


ImageServer::ImageServer(ImageResidency& residency, LoadImage load, ImageServerOptions opts)
    : impl_(std::make_unique<Impl>(residency, std::move(load), std::move(opts)))
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
