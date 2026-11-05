#pragma once

#include <Eigen/Dense>
#include <vector>

#include "core/model.hpp"
#include "linalg/view.hpp"

namespace saddle::hess {

struct NoiseEstimate {
  double epsilon;   // estimated gradient noise, Eh/bohr (per component RMS)
  int order;        // finite-difference order used
  int n_gradients;  // gradients evaluated
};

// Estimate gradient noise from samples g(x + k*delta*u), k = 0..n-1, using
// order-m forward differences. A smooth surface contributes ~delta^m g^(m),
// which vanishes; independent noise of RMS eps contributes variance
// eps^2 * C(2m, m). Spec §6 "Noise probe".
[[nodiscard]] NoiseEstimate estimate_noise_from_line(const std::vector<Eigen::VectorXd>& samples,
                                                     int order);

// h_opt = (3 eps / |E''''|)^(1/3)
[[nodiscard]] double optimal_step(double epsilon, double fourth_derivative);

// dk(h) = eps / h + |E''''| h^2 / 6
[[nodiscard]] double fd_error(double epsilon, double step, double fourth_derivative);

// Frequency (cm^-1) of a pure force-constant error dk (Eh/bohr^2) on mass m (amu).
[[nodiscard]] double noise_frequency_cm(double force_constant_error, double mass_amu);

template <ElectronicModel Model>
[[nodiscard]] NoiseEstimate probe_noise(Model& model, const Geometry& reference,
                                        const Eigen::VectorXd& direction, double delta,
                                        int order = 6, int n_samples = 16) {
  if (direction.size() != reference.dimension()) {
    throw ShapeError("noise probe direction does not match geometry dimension");
  }
  const double norm = direction.norm();
  if (!(norm > 0.0)) throw std::invalid_argument("noise probe direction must be nonzero");
  if (!(delta > 0.0)) throw std::invalid_argument("noise probe delta must be positive");
  if (order < 2 || n_samples < order + 2) {
    throw std::invalid_argument("noise probe needs order >= 2 and n_samples >= order + 2");
  }
  const Eigen::VectorXd u = direction / norm;
  std::vector<Eigen::VectorXd> samples;
  samples.reserve(static_cast<std::size_t>(n_samples));
  for (int k = 0; k < n_samples; ++k) {
    Geometry g = reference;
    g.coords += (k * delta) * u;
    samples.push_back(model.evaluate(g, WarmStart{}).gradient);
  }
  return estimate_noise_from_line(samples, order);
}

}  // namespace saddle::hess
