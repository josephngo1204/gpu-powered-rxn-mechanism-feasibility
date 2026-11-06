#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>

#include "core/model.hpp"
#include "hess/davidson.hpp"
#include "hess/vibrations.hpp"
#include "linalg/constants.hpp"
#include "linalg/view.hpp"

namespace saddle::hess {

struct ScreenOptions {
  int n_roots = 1;
  double step = 5e-3;
  double tolerance_cm = 1.0;
  int max_matvecs = 60;
};

struct ScreenResult {
  Eigen::VectorXd frequencies_cm;
  Eigen::MatrixXd modes_cartesian;
  int n_matvecs;
  int n_gradients;
  bool converged;
};

namespace detail {
inline double eigenvalue_tolerance(double tolerance_cm) {
  using constants::kAmuToElectronMass;
  using constants::kHartreeToWavenumber;
  constexpr double kReferenceFrequency = 10.0;
  return 2.0 * kReferenceFrequency * tolerance_cm * kAmuToElectronMass /
         (kHartreeToWavenumber * kHartreeToWavenumber);
}
}

template <ElectronicModel Model>
[[nodiscard]] ScreenResult partial_hessian_screen(Model& model, const Geometry& reference,
                                                  const Eigen::VectorXd& masses_amu,
                                                  const ScreenOptions& options,
                                                  const Eigen::VectorXd& guess_cartesian = {}) {
  const Eigen::Index n3 = reference.dimension();
  if (masses_amu.size() * 3 != n3) throw ShapeError("partial screen: masses do not match geometry");
  if (guess_cartesian.size() != 0 && guess_cartesian.size() != n3) {
    throw ShapeError("partial screen: guess does not match geometry");
  }
  if (!(options.step > 0.0)) throw std::invalid_argument("partial screen: step must be positive");

  const Eigen::MatrixXd internal =
      orthogonal_complement(external_basis(reference.coords, masses_amu));
  const Eigen::VectorXd inv_sqrt_m = inverse_sqrt_masses(masses_amu);
  const Eigen::VectorXd sqrt_m = inv_sqrt_m.cwiseInverse();

  int gradients = 0;
  auto grad_at = [&](const Eigen::VectorXd& displacement) {
    Geometry g = reference;
    g.coords += displacement;
    ++gradients;
    return model.evaluate(g, WarmStart{}).gradient;
  };
  auto matvec = [&](const Eigen::VectorXd& y) -> Eigen::VectorXd {
    const Eigen::VectorXd x = inv_sqrt_m.cwiseProduct(internal * y);
    const double norm = x.norm();
    const Eigen::VectorXd u = x / norm;
    const Eigen::VectorXd hu = (grad_at(options.step * u) - grad_at(-options.step * u)) / (2.0 * options.step);
    return internal.transpose() * inv_sqrt_m.cwiseProduct(hu * norm);
  };

  DavidsonOptions dopt;
  dopt.n_roots = options.n_roots;
  dopt.tolerance = detail::eigenvalue_tolerance(options.tolerance_cm);
  dopt.max_matvecs = options.max_matvecs;
  Eigen::MatrixXd guess;
  if (guess_cartesian.size() == n3) guess = internal.transpose() * sqrt_m.cwiseProduct(guess_cartesian);
  const auto d = davidson(matvec, internal.cols(), dopt, {}, guess);

  ScreenResult out;
  out.frequencies_cm.resize(d.eigenvalues.size());
  for (Eigen::Index k = 0; k < d.eigenvalues.size(); ++k) out.frequencies_cm[k] = eigenvalue_to_wavenumber(d.eigenvalues[k]);
  out.modes_cartesian = inv_sqrt_m.asDiagonal() * (internal * d.eigenvectors);
  for (Eigen::Index k = 0; k < out.modes_cartesian.cols(); ++k) out.modes_cartesian.col(k).normalize();
  out.n_matvecs = d.n_matvecs;
  out.n_gradients = gradients;
  out.converged = d.converged;
  return out;
}

}
