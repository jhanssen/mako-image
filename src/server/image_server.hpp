#pragma once

#include "image/engine.hpp"
#include "server/image_models.hpp"
#include "server/residency.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace mako::server {

using ImageResidency = Residency<image::Engine>;
// Loads a model with LoRAs merged.
using LoadImage = std::function<std::unique_ptr<image::Engine>(const ImageModel&, const image::Loras&)>;

struct ImageServerOptions {
    std::string host = "127.0.0.1";
    int port = 8081;
    std::vector<std::string> api_keys; // empty: requests only from this machine (and no other web page)
    // A directory of model directories (read on each request; see ImageModel), and the one for requests that name
    // none (a name there, or a model directory alone; default: the first).
    std::string image_models, image_model;
    uint32_t max_waiting = 4; // requests waiting beyond the one generating; more get HTTP 503
};

// OpenAI's image API: POST /v1/images/generations, GET /v1/images/models, GET /v1/images/loras, and /health and
// /v1/status.
class ImageServer {
public:
    ImageServer(ImageResidency& residency, LoadImage load, ImageServerOptions opts);
    ~ImageServer();
    // Blocks until stop(); false if it could not listen.
    bool run();
    void stop();
    // The port listened on (after run() started; the given one, or the one chosen for port 0).
    int port() const;

    // The model a request names ("" for the default). Throws std::invalid_argument when there is none such.
    ImageModel model(const std::string& name) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mako::server
