// The image server's HTTP side without a model: health and status, access checks, request validation, and a model
// that cannot be found.
//   image_server_test
#include "server/image_server.hpp"

#include <httplib.h>

#include <chrono>
#include <print>
#include <stdexcept>
#include <string>
#include <thread>

using namespace std::chrono_literals;

namespace {

int failures = 0;

void check(bool ok, const std::string& what)
{
    std::println("  {}  {}", ok ? "ok  " : "FAIL", what);
    failures += !ok;
}

} // namespace

int main()
{
    mako::server::ImageResidency residency(nullptr);
    mako::server::ImageServerOptions opts;
    opts.port = 0;
    opts.image_model = "/nonexistent/model";
    mako::server::ImageServer server(
        residency, [](const mako::server::ImageModel&, const mako::image::Loras&) -> std::unique_ptr<mako::image::Engine> {
            throw std::runtime_error("no models in this test");
        },
        opts);
    std::thread t([&] { server.run(); });
    for (int i = 0; i < 200 && !server.port(); ++i)
        std::this_thread::sleep_for(10ms);
    httplib::Client c("127.0.0.1", server.port());

    auto r = c.Get("/health");
    check(r && r->status == 200, "/health");
    r = c.Get("/v1/status");
    check(r && r->status == 200 && r->body.contains(R"("state":"none")"), "/v1/status: nothing loaded");
    r = c.Get("/v1/images/models");
    check(r && r->status == 200 && r->body.contains(R"("data":[])"), "/v1/images/models: none to list");
    r = c.Get("/v1/status", httplib::Headers{{"Host", "evil.example.com"}});
    check(r && r->status == 403, "another Host is refused without a key");
    r = c.Get("/v1/status", httplib::Headers{{"Origin", "https://evil.example.com"}});
    check(r && r->status == 403, "another web page is refused without a key");
    const auto post = [&](const std::string& body) { return c.Post("/v1/images/generations", body, "application/json"); };
    r = post("not json");
    check(r && r->status == 400, "a body that is not JSON");
    r = post(R"({"size": "1024x1024"})");
    check(r && r->status == 400 && r->body.contains("prompt is required"), "no prompt");
    r = post(R"({"prompt": "a cat", "size": "100x100"})");
    check(r && r->status == 400 && r->body.contains("size must be"), "a size the models cannot make");
    r = post(R"({"prompt": "a cat", "n": 9})");
    check(r && r->status == 400 && r->body.contains("n must be"), "too many images");
    r = post(std::string(R"({"prompt": ")") + std::string(20000, 'a') + R"("})");
    check(r && r->status == 400 && r->body.contains("longer than"), "a prompt too long");
    r = post(R"({"prompt": "a cat"})");
    check(r && r->status == 400 && r->body.contains("error"), "the default model's directory missing: an error");
    server.stop();
    t.join();
    std::println("{}", failures ? "FAILED" : "all passed");
    return failures ? 1 : 0;
}
