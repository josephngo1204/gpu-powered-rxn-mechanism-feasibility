#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <vector>

#include "core/model.hpp"
#include "hess/fd_hessian.hpp"
#include "search/rfo.hpp"

namespace saddle::search {

enum class InitialHessian { Exact, Identity };

struct OptimizerOptions {
  int max_iterations = 200;
  double gtol_rms = 1e-5;  // spec §5 G1, Eh/bohr
  double gtol_max = 1.5e-5;
  double trust_radius = 0.1;
  double max_trust_radius = 0.3;
  double min_trust_radius = 1e-5;
  InitialHessian initial_hessian = InitialHessian::Exact;
  double identity_scale = 1.0;
  double fd_step = 1e-4;
};

struct OptimizationResult {
  Geometry geometry;
  Evaluation evaluation;
  Eigen::MatrixXd hessian;  // final (updated) Hessian
  int iterations = 0;
  bool converged = false;
  int n_gradients = 0;
  int n_hessians = 0;
  std::vector<double> energies;
  std::vector<double> step_norms;
};

[[nodiscard]] inline bool gradient_converged(const Eigen::VectorXd& g, const OptimizerOptions& o) {
  return rms(g) <= o.gtol_rms && g.cwiseAbs().maxCoeff() <= o.gtol_max;
}

namespace detail {

template <ElectronicModel Model>
Eigen::MatrixXd initial_hessian(Model& model, const Geometry& x, const OptimizerOptions& o,
                                OptimizationResult& r) {
  if (o.initial_hessian == InitialHessian::Exact) {
    r.n_gradients += static_cast<int>(2 * x.dimension());
    r.n_hessians += 1;
    return hess::estimate_hessian(model, x, o.fd_step).hessian;
  }
  return o.identity_scale * Eigen::MatrixXd::Identity(x.dimension(), x.dimension());
}

inline double update_trust(double trust, double ratio, double step_norm, const OptimizerOptions& o) {
  if (ratio < 0.25) return std::max(o.min_trust_radius, 0.5 * trust);
  if (ratio > 0.75 && step_norm >= 0.95 * trust) return std::min(o.max_trust_radius, 2.0 * trust);
  return trust;
}

}  // namespace detail

// RS-RFO minimization with BFGS updates. Translations and rotations are
// removed for molecular geometries. Uphill steps are rejected and the trust
// radius shrinks.
template <ElectronicModel Model>
[[nodiscard]] OptimizationResult minimize(Model& model, const Geometry& guess,
                                          const OptimizerOptions& o) {
  OptimizationResult r;
  r.geometry = guess;
  r.evaluation = model.evaluate(r.geometry, WarmStart{});
  r.n_gradients += 1;
  r.energies.push_back(r.evaluation.energy.value());
  Eigen::MatrixXd h = detail::initial_hessian(model, r.geometry, o, r);
  double trust = o.trust_radius;

  for (; r.iterations < o.max_iterations; ++r.iterations) {
    if (gradient_converged(r.evaluation.gradient, o)) break;
    const Eigen::MatrixXd basis = internal_basis(r.geometry);
    const RfoStep rfo = rs_rfo_minimize_step(basis.transpose() * h * basis,
                                             basis.transpose() * r.evaluation.gradient, trust);
    const Eigen::VectorXd step = basis * rfo.step;
    Geometry trial = r.geometry;
    trial.coords += step;
    Evaluation e = model.evaluate(trial, WarmStart{});
    r.n_gradients += 1;
    r.step_norms.push_back(step.norm());

    h = bfgs_update(h, step, e.gradient - r.evaluation.gradient);
    const double actual = e.energy.value() - r.evaluation.energy.value();
    const double ratio = rfo.predicted_change != 0.0 ? actual / rfo.predicted_change : 1.0;
    trust = detail::update_trust(trust, ratio, step.norm(), o);
    if (actual > 0.0 && !gradient_converged(e.gradient, o)) continue;  // reject uphill step
    r.geometry = std::move(trial);
    r.evaluation = std::move(e);
    r.energies.push_back(r.evaluation.energy.value());
  }
  r.converged = gradient_converged(r.evaluation.gradient, o);
  r.hessian = std::move(h);
  return r;
}

}  // namespace saddle::search
