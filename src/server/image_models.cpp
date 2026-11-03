#include "server/image_models.hpp"

#include "log/log.hpp"
#include "safetensors/safetensors.hpp"

#include <glaze/glaze.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <optional>
#include <stdexcept>

namespace mako::server {

namespace {

namespace fs = std::filesystem;

std::string resolve(const fs::path& base, const std::string& p)
{
    if (p.starts_with("~/"))
        if (const char* home = std::getenv("HOME"))
            return (fs::path(home) / p.substr(2)).lexically_normal().string();
    const fs::path q(p);
    return (q.is_absolute() ? q : base / q).lexically_normal().string();
}

bool holds_model(const fs::path& d)
{
    return fs::exists(d / "model_index.json") || fs::exists(d / "mako.json") || fs::is_directory(d / "split_files") ||
           fs::is_directory(d / "diffusion_models");
}

} // namespace

std::optional<std::vector<uint32_t>> parse_skip_blocks(std::string_view json)
{
    glz::generic v;
    if (glz::read_json(v, json) || !v.is_array())
        return std::nullopt;
    std::vector<uint32_t> out;
    for (const auto& b : v.get_array()) {
        if (!b.is_number() || b.get_number() < 0 || b.get_number() > 1000 ||
            b.get_number() != std::floor(b.get_number()))
            return std::nullopt;
        out.push_back(uint32_t(b.get_number()));
    }
    return out;
}

ImageModel load_image_model(const std::string& dir, const std::string& name)
{
    const fs::path base = fs::absolute(fs::path(dir)).lexically_normal();
    ImageModel m;
    m.name = name;
    m.dir = base.string();
    std::string lora_dir = (base / "loras").string();
    const fs::path cfg = base / "mako.json";
    if (fs::exists(cfg)) {
        std::ifstream in(cfg, std::ios::binary);
        const std::string text{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
        glz::generic j;
        if (glz::read_json(j, text) || !j.is_object())
            throw std::runtime_error(cfg.string() + ": not a JSON object");
        auto str = [&](const char* key) -> std::optional<std::string> {
            if (!j.contains(key))
                return std::nullopt;
            if (!j[key].is_string())
                throw std::runtime_error(std::format("{}: {} must be a string", cfg.string(), key));
            return j[key].get_string();
        };
        auto num = [&](const char* key) -> std::optional<double> {
            if (!j.contains(key))
                return std::nullopt;
            if (!j[key].is_number())
                throw std::runtime_error(std::format("{}: {} must be a number", cfg.string(), key));
            return j[key].get_number();
        };
        if (auto v = str("model"))
            m.dir = resolve(base, *v);
        if (auto v = str("transformer"))
            m.options.transformer = resolve(base, *v);
        if (auto v = str("text_encoder"))
            m.options.text_encoder = resolve(base, *v);
        if (auto v = str("adapter"))
            m.options.adapter = resolve(base, *v);
        if (auto v = str("control"))
            m.options.control = resolve(base, *v);
        if (auto v = str("description"))
            m.description = *v;
        if (auto v = str("prompting"))
            m.prompting = *v;
        if (auto v = str("lora_dir"))
            lora_dir = resolve(base, *v);
        if (auto v = num("steps")) {
            if (*v < 0 || *v > 100 || *v != double(uint32_t(*v)))
                throw std::runtime_error(cfg.string() + ": steps must be a whole number from 0 to 100");
            m.steps = uint32_t(*v);
        }
        if (auto v = num("terminal"))
            m.terminal = float(*v);
        if (auto v = num("guidance"))
            m.guidance = float(*v);
        if (auto v = str("sampler")) {
            m.sampler = image::parse_sampler(*v);
            if (!m.sampler)
                throw std::runtime_error(cfg.string() + ": sampler must be euler, euler_a or er_sde");
        }
        if (j.contains("skip_blocks")) {
            std::string text;
            (void)glz::write_json(j["skip_blocks"], text);
            m.skip_blocks = parse_skip_blocks(text);
            if (!m.skip_blocks)
                throw std::runtime_error(cfg.string() + ": skip_blocks must be an array of block numbers");
        }
        if (j.contains("loras")) {
            if (!j["loras"].is_object())
                throw std::runtime_error(cfg.string() + ": loras must be an object of name: scale");
            for (const auto& [n, s] : j["loras"].get_object()) {
                image::Lora l;
                if (s.is_number()) {
                    l.scale = float(s.get_number());
                } else if (s.is_object() && (!s.contains("scale") || s["scale"].is_number()) &&
                           (!s.contains("merge") || s["merge"].is_boolean())) {
                    if (s.contains("scale"))
                        l.scale = float(s["scale"].get_number());
                    if (s.contains("merge"))
                        l.merge = s["merge"].get_boolean();
                } else {
                    throw std::runtime_error(std::format(
                        "{}: LoRA {} must be a scale or {{\"scale\": F, \"merge\": true|false}}", cfg.string(), n));
                }
                const fs::path f = fs::path(lora_dir) / (n.ends_with(".safetensors") ? n : n + ".safetensors");
                if (!fs::exists(f))
                    throw std::runtime_error(std::format("{}: no LoRA {}", cfg.string(), f.string()));
                l.path = f.string();
                m.loras.push_back(std::move(l));
            }
        }
    }
    for (const std::string* p : {&m.options.transformer, &m.options.text_encoder, &m.options.adapter,
                                 &m.options.control})
        if (!p->empty() && !fs::exists(*p))
            throw std::runtime_error(std::format("{}: {} does not exist", name, *p));
    if (!fs::is_directory(m.dir))
        throw std::runtime_error(std::format("{}: {} is not a directory", name, m.dir));
    if (fs::is_directory(lora_dir))
        m.lora_dir = lora_dir;
    m.family = image::Engine::family(m.dir);
    if (m.family.empty())
        throw std::runtime_error(std::format("{}: {} holds no image model mako knows", name, m.dir));
    if (m.sampler == image::Sampler::euler_ancestral && !image::Engine::predicts_noise(m.family))
        throw std::runtime_error(std::format("{}: euler_a needs a model that predicts the noise", name));
    if (!m.options.adapter.empty() && ((m.guidance && *m.guidance != 1.0f) || m.sampler))
        throw std::runtime_error(std::format("{}: an adapter runs its own sampler, without guidance", name));
    if (!m.options.adapter.empty() && m.family != "MOSS-Image")
        throw std::runtime_error(std::format("{}: adapters are for MOSS-Image", name));
    if (!m.options.control.empty() && !image::Engine::takes_control(m.family))
        throw std::runtime_error(std::format("{}: control adapters are for MOSS-Image", name));
    if (!m.options.control.empty() && !m.options.adapter.empty())
        throw std::runtime_error(std::format("{}: an adapter's sampler and a control adapter do not go together", name));
    for (const auto& l : m.loras)
        if (l.merge == false && !image::Engine::keeps_loras_unmerged(m.family))
            throw std::runtime_error(std::format("{}: {} keeps no LoRA unmerged", name, m.family));
    if (m.skip_blocks && !m.skip_blocks->empty() && !image::Engine::skips_blocks(m.family))
        throw std::runtime_error(std::format("{}: {} has no blocks to skip", name, m.family));
    return m;
}

std::vector<ImageModel> image_models(const std::string& root)
{
    std::vector<ImageModel> out;
    std::vector<fs::path> dirs;
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(root, ec))
        if (e.is_directory(ec) && holds_model(e.path()))
            dirs.push_back(e.path());
    std::ranges::sort(dirs);
    for (const auto& d : dirs) {
        try {
            out.push_back(load_image_model(d.string(), d.filename().string()));
        } catch (const std::exception& e) {
            log::warn("image model {}: {}", d.filename().string(), e.what());
        }
    }
    return out;
}

std::vector<LoraChoice> lora_choices(const image::Loras& defaults, const std::string& dir)
{
    namespace fs = std::filesystem;
    std::vector<LoraChoice> out;
    auto known = [&](const std::string& name) { return std::ranges::contains(out, name, &LoraChoice::name); };
    for (const auto& l : defaults) {
        const std::string name = fs::path(l.path).stem().string();
        if (!known(name))
            out.push_back({name, l.path, l.scale, l.merge});
    }
    if (dir.empty())
        return out;
    std::vector<fs::path> files;
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(dir, ec))
        if (e.path().extension() == ".safetensors" && e.is_regular_file(ec))
            files.push_back(e.path());
    std::ranges::sort(files);
    for (const auto& f : files)
        if (!known(f.stem().string()))
            out.push_back({f.stem().string(), f.string(), std::nullopt, std::nullopt});
    return out;
}

LoraNotes lora_notes(const std::string& path)
{
    LoraNotes n;
    const auto sidecar = std::filesystem::path(path).replace_extension(".json");
    if (std::ifstream in(sidecar); in) {
        const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        glz::generic v;
        if (glz::read_json(v, text) || !v.is_object()) {
            log::warn("{}: not a JSON object; its notes left out", sidecar.string());
        } else {
            if (v.contains("trigger") && v["trigger"].is_string())
                n.trigger = v["trigger"].get_string();
            if (v.contains("description") && v["description"].is_string())
                n.description = v["description"].get_string();
            if (v.contains("scale_range")) {
                auto& r = v["scale_range"];
                if (r.is_array() && r.get_array().size() == 2 && r[0].is_number() && r[1].is_number() &&
                    r[0].get_number() <= r[1].get_number())
                    n.scale_range = {float(r[0].get_number()), float(r[1].get_number())};
                else
                    log::warn("{}: scale_range is not [low, high]; left out", sidecar.string());
            }
        }
    }
    if (!n.trigger.empty() && !n.description.empty())
        return n;
    auto f = safetensors::File::open(path);
    if (!f)
        return n;
    const auto& m = f->metadata();
    auto first = [&](std::initializer_list<const char*> keys) -> std::string {
        for (const char* k : keys)
            if (auto it = m.find(k); it != m.end() && !it->second.empty())
                return it->second;
        return {};
    };
    if (n.trigger.empty())
        n.trigger = first({"trigger_word", "modelspec.trigger_phrase"});
    if (n.description.empty())
        n.description = first({"modelspec.description"});
    return n;
}

image::Loras resolve_loras(std::string_view json, const std::vector<LoraChoice>& choices)
{
    glz::generic v;
    if (glz::read_json(v, json) || !v.is_array())
        throw std::invalid_argument("an array of {\"name\", \"scale\"} objects");
    image::Loras out;
    std::vector<std::string> names;
    for (auto& e : v.get_array()) {
        if (!e.is_object() || !e.contains("name") || !e["name"].is_string())
            throw std::invalid_argument("each entry needs a \"name\"");
        std::string name = e["name"].get_string();
        if (name.ends_with(".safetensors"))
            name.resize(name.size() - std::string_view(".safetensors").size());
        const auto it = std::ranges::find(choices, name, &LoraChoice::name);
        if (it == choices.end())
            throw std::invalid_argument(std::format("no LoRA named \"{}\" (GET /v1/images/loras lists them)", name));
        if (std::ranges::contains(names, name))
            throw std::invalid_argument(std::format("\"{}\" named twice", name));
        float scale = 1.0f;
        if (e.contains("scale") && !e["scale"].is_null()) {
            if (!e["scale"].is_number() || !std::isfinite(e["scale"].get_number()) ||
                std::abs(e["scale"].get_number()) > 10)
                throw std::invalid_argument(std::format("\"{}\": scale must be a number from -10 to 10", name));
            scale = float(e["scale"].get_number());
        }
        std::optional<bool> merge = it->merge;
        if (e.contains("merge") && !e["merge"].is_null()) {
            if (!e["merge"].is_boolean())
                throw std::invalid_argument(std::format("\"{}\": merge must be true or false", name));
            merge = e["merge"].get_boolean();
        }
        names.push_back(name);
        out.push_back({it->path, scale, merge});
    }
    return out;
}


} // namespace mako::server
