#pragma once

#include <Eigen/Dense>
#include <vector>

#include "core/model.hpp"
#include "linalg/view.hpp"

namespace saddle::hess {

struct NoiseEstimate {
  double epsilon;
  int order;
  int n_gradients;
};

[[nodiscard]] NoiseEstimate estimate_noise_from_line(const std::vector<Eigen::VectorXd>& samples,
                                                     int order);

[[nodiscard]] double optimal_step(double epsilon, double fourth_derivative);

[[nodiscard]] double fd_error(double epsilon, double step, double fourth_derivative);

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

}
