#include "image/er_sde.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mako::image {

namespace {

double scaler(double x) { return x * (std::exp(std::pow(x, 0.3)) + 10.0); }

constexpr int kPoints = 200; // the integrals' sample points

} // namespace

ErSde::ErSde(std::span<const float> sigmas, bool flow, float top)
    : sigmas_(sigmas.begin(), sigmas.end())
{
    if (sigmas_.size() < 2)
        throw std::runtime_error("ER-SDE: no steps");
    if (flow && sigmas_[0] >= 1)
        sigmas_[0] = top;
    for (double s : sigmas_)
        lambdas_.push_back(flow ? s / (1 - s) : s);
}

void ErSde::step(uint32_t i, std::span<float> x, std::span<const float> denoised, std::span<const float> noise,
                 float s_noise)
{
    const size_t n = x.size();
    if (i + 1 >= sigmas_.size() || denoised.size() != n)
        throw std::runtime_error("ER-SDE: step or size out of range");
    const uint32_t stage = std::min(3u, i + 1);
    if (sigmas_[i + 1] == 0) {
        std::copy(denoised.begin(), denoised.end(), x.begin());
    } else {
        if (noise.size() != n)
            throw std::runtime_error("ER-SDE: noise size");
        const double ls = lambdas_[i], lt = lambdas_[i + 1];
        const double alpha_s = sigmas_[i] / ls, alpha_t = sigmas_[i + 1] / lt;
        const double r = scaler(lt) / scaler(ls);
        const double c1 = alpha_t / alpha_s * r, c2 = alpha_t * (1 - r);
        const double dt = lt - ls, h = -dt / kPoints;
        double s = 0, s_u = 0;
        for (int k = 0; k < kPoints; ++k) {
            const double pos = lt + k * h, sp = scaler(pos);
            s += 1 / sp;
            s_u += (pos - ls) / sp;
        }
        s *= h;
        s_u *= h;
        const double c3 = alpha_t * (dt + s * scaler(lt)), c4 = alpha_t * (dt * dt / 2 + s_u * scaler(lt));
        const double d_den = stage >= 2 ? ls - lambdas_[i - 1] : 1, u_den = stage >= 3 ? (ls - lambdas_[i - 2]) / 2 : 1;
        const double cn = alpha_t * double(s_noise) * std::sqrt(std::max(0.0, lt * lt - ls * ls * r * r));
        std::vector<float> dd(stage >= 2 ? n : 0);
        for (size_t j = 0; j < n; ++j) {
            double v = c1 * x[j] + c2 * denoised[j];
            if (stage >= 2) {
                const double d = (double(denoised[j]) - old_[j]) / d_den;
                dd[j] = float(d);
                v += c3 * d;
                if (stage >= 3)
                    v += c4 * (d - old_d_[j]) / u_den;
            }
            if (s_noise > 0)
                v += cn * noise[j];
            x[j] = float(v);
        }
        if (stage >= 2)
            old_d_ = std::move(dd);
    }
    old_.assign(denoised.begin(), denoised.end());
}

} // namespace mako::image
