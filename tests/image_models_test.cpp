// Image model directories and LoRAs as requests name them: mako.json settings, LoRA choices and notes, a request's
// "loras". No model needed.
//   image_models_test
#include "server/image_models.hpp"

#include <glaze/glaze.hpp>

#include <filesystem>
#include <format>
#include <fstream>
#include <print>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

namespace {

int failures = 0;

void check(bool ok, const std::string& what)
{
    if (!ok) {
        ++failures;
        std::println("FAIL {}", what);
    }
}

void image_model_checks()
{
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / std::format("mako_models_test_{}", getpid());
    fs::create_directories(root / "a");
    fs::create_directories(root / "b" / "loras");
    fs::create_directories(root / "c");
    fs::create_directories(root / "notes");
    std::ofstream(root / "a" / "model_index.json") << R"({"_class_name": "Krea2Pipeline"})";
    std::ofstream(root / "b" / "loras" / "x.safetensors") << "x";
    std::ofstream(root / "b" / "mako.json")
        << R"({"model": "../a", "steps": 4, "guidance": 1.5, "terminal": 0, "loras": {"x": 0.5}, "description": "d", "prompting": "p"})";
    std::ofstream(root / "c" / "mako.json") << R"({"model": "../a", "loras": {"missing": 1}})";
    const auto list = mako::server::image_models(root.string());
    check(list.size() == 2 && list[0].name == "a" && list[1].name == "b",
          "image models: the directories holding one, a bad mako.json left out");
    if (list.size() == 2) {
        const auto& b = list[1];
        check(b.dir == fs::absolute(root / "a").lexically_normal().string() && b.steps == 4 && b.guidance == 1.5f &&
                  b.terminal == 0.0f && b.loras.size() == 1 && b.loras[0].second == 0.5f &&
                  b.lora_dir == fs::absolute(root / "b" / "loras").lexically_normal().string() &&
                  b.key() == list[0].key() && b.family == "Krea 2" && b.description == "d" && b.prompting == "p",
              "mako.json: the model it points at, its settings, its LoRAs from its loras/");
        check(list[0].lora_dir.empty() && list[0].loras.empty(), "a plain model directory: no LoRA directory");
    }
    fs::remove_all(root);
}

void lora_checks()
{
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / std::format("mako_lora_test_{}", getpid());
    fs::create_directories(dir);
    for (const char* f : {"b.safetensors", "a.safetensors", "style.safetensors", "notes.txt"})
        std::ofstream(dir / f) << "x";
    const mako::image::Loras defaults{{"/elsewhere/style.safetensors", 1.3f}};
    const auto choices = mako::server::lora_choices(defaults, dir.string());
    check(choices.size() == 3 && choices[0].name == "style" && choices[0].path == "/elsewhere/style.safetensors" &&
              choices[0].scale == 1.3f && choices[1].name == "a" && !choices[1].scale && choices[2].name == "b",
          "LoRA choices: start-up ones first (their name wins), then the directory's by name");
    using mako::server::resolve_loras;
    const auto got = resolve_loras(R"([{"name":"b","scale":-0.5},{"name":"style.safetensors"}])", choices);
    check(got == mako::image::Loras{{(dir / "b.safetensors").string(), -0.5f}, {"/elsewhere/style.safetensors", 1.0f}},
          "a request's LoRAs: files and scales in order, scale 1 by default, names with or without .safetensors");
    check(resolve_loras("[]", choices).empty(), "an empty list: no LoRAs");
    for (const char* bad : {R"({"name":"a"})", R"([{"name":"../a"}])", R"([{"name":"../a.safetensors"}])",
                            R"([{"name":"notes"}])", R"([{"name":"a"},{"name":"a.safetensors"}])", R"([{"scale":1}])",
                            R"([{"name":"a","scale":"1"}])", R"([{"name":"a","scale":11}])",
                            R"([{"name":"a"},{"name":"a"}])"}) {
        bool threw = false;
        try {
            resolve_loras(bad, choices);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        check(threw, std::format("LoRAs refused: {}", bad));
    }
    check(mako::server::lora_choices(defaults, "").size() == 1, "no directory: the start-up LoRAs only");
    {
        const std::string header =
            R"({"__metadata__":{"trigger_word":"from metadata","modelspec.description":"meta description"}})";
        std::ofstream f(dir / "c.safetensors", std::ios::binary);
        const uint64_t n = header.size();
        f.write(reinterpret_cast<const char*>(&n), 8);
        f << header;
    }
    std::ofstream(dir / "a.json") << R"({"trigger":"Next Scene:","description":"a later moment","scale_range":[0.6,1.2]})";
    std::ofstream(dir / "c.json") << R"({"description":"sidecar description","scale_range":[2,1]})";
    std::ofstream(dir / "b.json") << "not json";
    using mako::server::lora_notes;
    const auto a = lora_notes((dir / "a.safetensors").string()), c = lora_notes((dir / "c.safetensors").string()),
               b = lora_notes((dir / "b.safetensors").string());
    check(a.trigger == "Next Scene:" && a.description == "a later moment" &&
              a.scale_range == std::pair(0.6f, 1.2f),
          "LoRA notes from a sidecar");
    check(c.trigger == "from metadata" && c.description == "sidecar description" && !c.scale_range,
          "LoRA notes: the sidecar's fields first, the metadata's for the rest");
    check(b.trigger.empty() && b.description.empty(), "LoRA notes: a bad sidecar and a bad file give none");
    fs::remove_all(dir);
}

} // namespace

int main()
{
    lora_checks();
    image_model_checks();
    std::println("{}", failures ? "FAILED" : "image_models: all cases pass");
    return failures ? 1 : 0;
}
