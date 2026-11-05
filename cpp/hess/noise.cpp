#include "hess/noise.hpp"

#include <cmath>
#include <stdexcept>

#include "linalg/constants.hpp"

namespace saddle::hess {

namespace {

double binomial(int n, int k) {
  double out = 1.0;
  for (int i = 1; i <= k; ++i) out = out * (n - k + i) / i;
  return out;
}

}  // namespace

NoiseEstimate estimate_noise_from_line(const std::vector<Eigen::VectorXd>& samples, int order) {
  const auto n = static_cast<int>(samples.size());
  if (order < 2 || n < order + 2) {
    throw std::invalid_argument("noise estimate needs order >= 2 and at least order + 2 samples");
  }
  const Eigen::Index dim = samples.front().size();
  for (const auto& s : samples) {
    if (s.size() != dim) throw ShapeError("noise samples have inconsistent lengths");
  }
  // Coefficients of the order-m forward difference: (-1)^(m-j) C(m, j).
  std::vector<double> coeff(static_cast<std::size_t>(order + 1));
  for (int j = 0; j <= order; ++j) {
    coeff[static_cast<std::size_t>(j)] = ((order - j) % 2 == 0 ? 1.0 : -1.0) * binomial(order, j);
  }
  double sum_sq = 0.0;
  long count = 0;
  for (int start = 0; start + order < n; ++start) {
    Eigen::VectorXd diff = Eigen::VectorXd::Zero(dim);
    for (int j = 0; j <= order; ++j) {
      diff += coeff[static_cast<std::size_t>(j)] * samples[static_cast<std::size_t>(start + j)];
    }
    sum_sq += diff.squaredNorm();
    count += dim;
  }
  const double variance = sum_sq / static_cast<double>(count) / binomial(2 * order, order);
  return NoiseEstimate{std::sqrt(variance), order, n};
}

double optimal_step(double epsilon, double fourth_derivative) {
  if (!(epsilon > 0.0) || !(fourth_derivative > 0.0)) {
    throw std::invalid_argument("optimal_step needs positive noise and fourth derivative");
  }
  return std::cbrt(3.0 * epsilon / fourth_derivative);
}

double fd_error(double epsilon, double step, double fourth_derivative) {
  if (!(step > 0.0)) throw std::invalid_argument("fd_error needs a positive step");
  return epsilon / step + fourth_derivative * step * step / 6.0;
}

double noise_frequency_cm(double force_constant_error, double mass_amu) {
  if (!(mass_amu > 0.0)) throw std::invalid_argument("mass must be positive");
  return std::sqrt(std::abs(force_constant_error) / (mass_amu * constants::kAmuToElectronMass)) *
         constants::kHartreeToWavenumber;
}

}  // namespace saddle::hess
