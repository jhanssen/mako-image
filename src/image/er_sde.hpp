#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace mako::image {

// The extended reverse-time SDE solver (ER-SDE, three stages): each step moves the latents to the next noise level from
// the model's estimate of the image and the estimates' differences over the last two steps, then adds fresh noise.
// Flow models' latents are (1 - sigma) image + sigma noise, the others' image + sigma noise.
class ErSde {
public:
    // sigmas: the schedule, ending at 0. top: a flow model's level for a first sigma of 1 (where the noise has no
    // image left to scale), its level a hundredth of a percent of the way down.
    ErSde(std::span<const float> sigmas, bool flow, float top);
    // x at level i to level i + 1, given the image the model sees in it; noise: x.size() unit normals (unused at the
    // last step), scaled by s_noise.
    void step(uint32_t i, std::span<float> x, std::span<const float> denoised, std::span<const float> noise,
              float s_noise = 1.0f);
    // The level step i runs the model at: the schedule's, the first moved to top.
    float sigma(uint32_t i) const { return float(sigmas_.at(i)); }

private:
    std::vector<double> sigmas_, lambdas_; // lambda: sigma over the image's scale
    std::vector<float> old_, old_d_;
};

} // namespace mako::image
