// OpenAI's image API for Qwen-Image 2.1, Krea 2, MOSS-Image and Anima.
//   mako_image_serve [--image-models DIR] [--image-model NAME|DIR] [--upscalers DIR] [--host H] [--port N]
//                    [--api-key KEY[,KEY...]] [--api-key-file FILE] [--max-waiting N] [--llm URL] [--llm-api-key KEY]
// --image-models: a directory of image model directories (each its files as published, or a mako.json pointing at
// them with its settings and LoRAs: server::ImageModel); a request's "model" picks one (GET /v1/images/models lists
// them). --image-model: the one for requests that name none (a name there, or a model directory; default: the first).
// A model loads with the first request for it and stays loaded; another model or other LoRAs replace it.
// --upscalers: a directory of ESRGAN upscalers, NAME.safetensors each (tools/esrgan_to_safetensors.py), for
// POST /v1/images/upscales and a generation's "upscale" (default: --image-models' upscalers/ when it exists).
// --llm: a mako_serve on this machine whose memory the models borrow (its URL, e.g. http://127.0.0.1:8080): before a
// model loads the LLM is unloaded, and when a chat request comes the model is freed and the LLM loads again. Without
// it the models' memory is their own (another machine, or one with room for both).
// Environment variables MAKO_IMAGE_<OPTION> (e.g. MAKO_IMAGE_PORT) stand for the options.
#include "app/cli.hpp"
#include "image/engine.hpp"
#include "image/esrgan.hpp"
#include "log/log.hpp"
#include "memory/borrower.hpp"
#include "mtl/context.hpp"
#include "server/image_models.hpp"
#include "server/image_server.hpp"

#include <cctype>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <memory>
#include <print>
#include <sstream>
#include <string>
#include <thread>

#include <pthread.h>
#include <unistd.h>

int main(int argc, char** argv)
{
    mako::server::ImageServerOptions so;
    std::string llm, llm_key;
    auto add_keys = [&](std::string_view list, char sep) {
        const size_t before = so.api_keys.size();
        for (size_t at = 0; at <= list.size();) {
            const size_t end = std::min(list.find(sep, at), list.size());
            std::string_view k = list.substr(at, end - at);
            while (!k.empty() && std::isspace(static_cast<unsigned char>(k.front())))
                k.remove_prefix(1);
            while (!k.empty() && std::isspace(static_cast<unsigned char>(k.back())))
                k.remove_suffix(1);
            if (!k.empty())
                so.api_keys.emplace_back(k);
            at = end + 1;
        }
        return so.api_keys.size() > before;
    };
    mako::app::Cli("mako_image_serve", "", 0, 0)
        .env_prefix("MAKO_IMAGE_")
        .option("--image-models", "DIR", "a directory of image model directories", so.image_models)
        .option("--image-model", "NAME|DIR", "the model for requests that name none (default: the first)",
                so.image_model)
        .option("--upscalers", "DIR", "a directory of ESRGAN upscalers (default: --image-models' upscalers/)",
                so.upscalers)
        .option("--host", "H", "address to listen on (default 127.0.0.1; others need an API key)", so.host)
        .option("--port", "N", "port to listen on (default 8082)", so.port)
        .option("--api-key", "KEY[,KEY...]", "keys requests must carry (Bearer or x-api-key)",
                [&](const std::string& v) { add_keys(v, ','); })
        .secret()
        .option("--api-key-file", "FILE", "keys requests must carry, one a line",
                [&](const std::string& v) {
                    std::ifstream in(v);
                    std::stringstream text;
                    text << in.rdbuf();
                    if (!in || !add_keys(text.str(), '\n'))
                        throw std::invalid_argument("names no readable key file");
                })
        .shared_env("--api-key")
        .option("--max-waiting", "N", "requests that may wait before new ones are refused", so.max_waiting)
        .option("--llm", "URL", "a mako_serve on this machine to borrow memory from (its LLM unloads meanwhile)", llm)
        .option("--llm-api-key", "KEY", "the key that mako_serve wants", llm_key)
        .secret()
        .parse(argc, argv);

    sigset_t stop_signals;
    sigemptyset(&stop_signals);
    sigaddset(&stop_signals, SIGINT);
    sigaddset(&stop_signals, SIGTERM);
    pthread_sigmask(SIG_BLOCK, &stop_signals, nullptr);
    for (const auto& k : so.api_keys)
        if (k.size() < 16)
            mako::log::warn("an API key of {} characters is easy to guess (16 or more; e.g. openssl rand -hex 24)",
                            k.size());
    if (so.host != "127.0.0.1" && so.host != "localhost" && so.host != "::1" && so.api_keys.empty()) {
        std::println(stderr, "--host {} serves beyond this machine: give an --api-key or --api-key-file", so.host);
        return 2;
    }
    if (so.image_models.empty() && so.image_model.empty()) {
        std::println(stderr, "give --image-models DIR or --image-model DIR");
        return 2;
    }
    if (!so.image_models.empty() && !std::filesystem::is_directory(so.image_models)) {
        std::println(stderr, "--image-models {}: not a directory", so.image_models);
        return 2;
    }
    if (so.image_models.empty() && !std::filesystem::is_directory(so.image_model)) {
        std::println(stderr, "--image-model {}: not a directory (or give --image-models)", so.image_model);
        return 2;
    }
    if (so.upscalers.empty() && !so.image_models.empty() &&
        std::filesystem::is_directory(std::filesystem::path(so.image_models) / "upscalers"))
        so.upscalers = (std::filesystem::path(so.image_models) / "upscalers").string();
    if (!so.upscalers.empty() && !std::filesystem::is_directory(so.upscalers)) {
        std::println(stderr, "--upscalers {}: not a directory", so.upscalers);
        return 2;
    }

    mako::mtl::Context ctx;
    std::unique_ptr<mako::memory::Borrower> borrower;
    if (!llm.empty())
        borrower = std::make_unique<mako::memory::Borrower>(llm, llm_key, std::format("mako_image_serve :{}", so.port));
    mako::server::ImageResidency residency(std::move(borrower));
    const auto load = [&](const mako::server::ImageModel& m, const mako::image::Loras& loras) {
        auto e = mako::image::Engine::open(m.dir, ctx, m.options);
        for (const auto& l : loras)
            e->add_lora(l);
        return e;
    };
    if (!so.image_models.empty())
        for (const auto& m : mako::server::image_models(so.image_models))
            mako::log::info("image model {}: {}{}", m.name, m.dir, m.loras.empty() ? "" : " (with LoRAs)");
    for (const auto& u : mako::server::upscaler_choices(so.upscalers))
        mako::log::info("upscaler {}: {} blocks, {}x", u.name, u.blocks, u.factor);
    const auto load_upscaler = [&](const std::string& path) { return std::make_unique<mako::image::Esrgan>(path, ctx); };
    mako::log::info("models load with the first request for them{}",
                    llm.empty() ? "" : std::format("; memory borrowed from the LLM at {}", llm));
    mako::server::ImageServer server(residency, load, so, load_upscaler);
    std::jthread stopper([&] {
        int sig = 0;
        sigwait(&stop_signals, &sig);
        server.stop();
    });
    mako::log::info("serving images on http://{}:{}/v1", so.host, so.port);
    const bool listened = server.run();
    if (!listened)
        mako::log::error("could not listen on {}:{}", so.host, so.port);
    ::kill(::getpid(), SIGTERM); // wakes the stopper if the server ended on its own
    return listened ? 0 : 1;
}
