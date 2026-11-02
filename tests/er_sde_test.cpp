// ER-SDE over a made-up model against the reference sampler's latents (flow and noise-predicting schedules, with and
// without noise; the noise made up too).
#include "image/er_sde.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

constexpr int kN = 6;

std::vector<float> run(std::vector<float> sigmas, bool flow, float top, float s_noise)
{
    mako::image::ErSde er(sigmas, flow, top);
    std::vector<float> x(kN), d(kN), noise(kN);
    for (int j = 0; j < kN; ++j)
        x[j] = float(std::sin(1.7 * j + 0.3)) * sigmas[0];
    for (uint32_t i = 0; i + 1 < sigmas.size(); ++i) {
        const float s = er.sigma(i);
        for (int j = 0; j < kN; ++j) {
            d[j] = std::tanh(x[j] * (0.6f + 0.2f * s)) + 0.1f * std::cos(float(j) + 3 * s);
            noise[j] = float(std::sin(0.9 * j + 1.3 * (i + 1)));
        }
        er.step(i, x, d, noise, s_noise);
    }
    return x;
}

bool check(const char* name, const std::vector<float>& got, const std::vector<float>& want)
{
    double err = 0, norm = 0;
    for (int j = 0; j < kN; ++j) {
        err += (got[j] - want[j]) * (got[j] - want[j]);
        norm += want[j] * want[j];
    }
    const double rel = std::sqrt(err / norm);
    std::printf("%s: rel error %.2e\n", name, rel);
    return rel < 1e-4;
}

} // namespace

int main()
{
    const std::vector<float> flow = {1.0f, 0.95f, 0.85f, 0.7f, 0.5f, 0.3f, 0.12f, 0.0f};
    const std::vector<float> eps = {14.6f, 8.0f, 4.0f, 2.0f, 1.0f, 0.4f, 0.1f, 0.0f};
    const float top = 0.9999799983998721f; // shift 5's level at 1e-4 of the way
    bool ok = check("flow", run(flow, true, top, 0),
                    {0.1173491f, -0.0196344f, -0.1478082f, -0.1345944f, 0.0104663f, 0.1382211f});
    ok &= check("flow, noise 1.1", run(flow, true, top, 1.1f),
                {0.2902035f, 0.0102337f, -0.2837239f, -0.3284311f, -0.1055133f, 0.2022891f});
    ok &= check("eps", run(eps, false, 0, 0),
                {0.4068552f, 0.3795589f, -0.4574052f, -0.4906585f, 0.3575438f, 0.4551862f});
    ok &= check("eps, noise 1", run(eps, false, 0, 1.0f),
                {0.4554704f, 0.3857054f, -0.4898459f, -0.5310282f, 0.3279795f, 0.4704102f});
    return ok ? 0 : 1;
}
