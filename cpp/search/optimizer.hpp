#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <utility>
#include <vector>

#include "core/model.hpp"
#include "hess/fd_hessian.hpp"
#include "linalg/view.hpp"
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

// One model evaluation, counted in the result.
template <ElectronicModel Model>
Evaluation evaluate_counted(Model& model, const Geometry& x, OptimizationResult& r) {
  r.n_gradients += 1;
  return model.evaluate(x, WarmStart{});
}

// Finite-difference Hessian at x (2 gradients per coordinate), counted in the result.
template <ElectronicModel Model>
Eigen::MatrixXd exact_hessian(Model& model, const Geometry& x, const OptimizerOptions& o,
                              OptimizationResult& r) {
  r.n_gradients += static_cast<int>(2 * x.dimension());
  r.n_hessians += 1;
  return hess::estimate_hessian(model, x, o.fd_step).hessian;
}

template <ElectronicModel Model>
Eigen::MatrixXd initial_hessian(Model& model, const Geometry& x, const OptimizerOptions& o,
                                OptimizationResult& r) {
  if (o.initial_hessian == InitialHessian::Exact) return exact_hessian(model, x, o, r);
  return o.identity_scale * Eigen::MatrixXd::Identity(x.dimension(), x.dimension());
}

// How well the quadratic model predicted the step: actual / predicted energy change.
inline double trust_ratio(double actual_change, double predicted_change) {
  return predicted_change != 0.0 ? actual_change / predicted_change : 1.0;
}

// Halve the trust radius after a poor prediction; double it after a good
// prediction from a step that used (nearly) the whole radius.
inline double update_trust(double trust, double ratio, double step_norm, const OptimizerOptions& o) {
  if (ratio < 0.25) return std::max(o.min_trust_radius, 0.5 * trust);
  if (ratio > 0.75 && step_norm >= 0.95 * trust) return std::min(o.max_trust_radius, 2.0 * trust);
  return trust;
}

struct ModeChoice {
  Eigen::Index index;
  double overlap;  // |cos| between the tracked direction and the chosen mode
};

// The mode (column of `modes`) most parallel to `tracked`. With nothing
// tracked yet, the lowest mode (column 0) is taken with full overlap.
inline ModeChoice best_overlap_mode(const Eigen::VectorXd& tracked, const Eigen::MatrixXd& modes) {
  if (tracked.size() == 0) return {0, 1.0};
  const Eigen::VectorXd t = tracked.normalized();
  ModeChoice best{0, -1.0};
  for (Eigen::Index k = 0; k < modes.cols(); ++k) {
    const double overlap = std::abs(t.dot(modes.col(k).normalized()));
    if (overlap > best.overlap) best = {k, overlap};
  }
  return best;
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
  r.evaluation = detail::evaluate_counted(model, r.geometry, r);
  r.energies.push_back(r.evaluation.energy.value());
  Eigen::MatrixXd h = detail::initial_hessian(model, r.geometry, o, r);
  double trust = o.trust_radius;

  for (; r.iterations < o.max_iterations; ++r.iterations) {
    if (gradient_converged(internal_gradient(r.geometry, r.evaluation.gradient), o)) break;
    // Step in internal coordinates (no translation/rotation), then back to Cartesian.
    const Eigen::MatrixXd basis = internal_basis(r.geometry);
    const RfoStep rfo = rs_rfo_minimize_step(basis.transpose() * h * basis,
                                             basis.transpose() * r.evaluation.gradient, trust);
    const Eigen::VectorXd step = basis * rfo.step;
    Geometry trial = r.geometry;
    trial.coords += step;
    Evaluation e = detail::evaluate_counted(model, trial, r);
    r.step_norms.push_back(step.norm());

    h = bfgs_update(h, step, e.gradient - r.evaluation.gradient);
    const double actual = e.energy.value() - r.evaluation.energy.value();
    trust = detail::update_trust(trust, detail::trust_ratio(actual, rfo.predicted_change),
                                 step.norm(), o);
    if (actual > 0.0 && !gradient_converged(internal_gradient(trial, e.gradient), o)) {
      continue;  // reject uphill step
    }
    r.geometry = std::move(trial);
    r.evaluation = std::move(e);
    r.energies.push_back(r.evaluation.energy.value());
  }
  r.converged = gradient_converged(internal_gradient(r.geometry, r.evaluation.gradient), o);
  r.hessian = std::move(h);
  return r;
}

struct SaddleOptions : OptimizerOptions {
  // Cartesian direction of the hypothesized reaction coordinate; when empty
  // the lowest Hessian mode is followed.
  Eigen::VectorXd follow_direction;
  double mode_overlap_threshold = 0.8;  // spec §6
  int max_hessian_recomputes = 10;
};

struct SaddleResult : OptimizationResult {
  int n_recomputes = 0;
  Eigen::VectorXd followed_mode;  // Cartesian, unit
  std::vector<double> mode_overlaps;
};

// RS-P-RFO first-order saddle search with Bofill updates. The followed mode
// is tracked by overlap; when the best overlap falls below the threshold the
// exact Hessian is recomputed. Every step is accepted (energy may rise).
template <ElectronicModel Model>
[[nodiscard]] SaddleResult saddle_search(Model& model, const Geometry& guess, const SaddleOptions& o) {
  SaddleResult r;
  r.geometry = guess;
  r.evaluation = detail::evaluate_counted(model, r.geometry, r);
  r.energies.push_back(r.evaluation.energy.value());
  Eigen::MatrixXd h = detail::initial_hessian(model, r.geometry, o, r);
  double trust = o.trust_radius;
  Eigen::VectorXd tracked = o.follow_direction;
  if (tracked.size() != 0 && tracked.size() != guess.dimension()) {
    throw ShapeError("follow_direction does not match geometry dimension");
  }

  for (; r.iterations < o.max_iterations; ++r.iterations) {
    if (gradient_converged(internal_gradient(r.geometry, r.evaluation.gradient), o)) break;
    const Eigen::MatrixXd basis = internal_basis(r.geometry);
    const Eigen::VectorXd g_internal = basis.transpose() * r.evaluation.gradient;
    Eigen::MatrixXd h_internal = basis.transpose() * h * basis;

    // Pick the mode to climb: the one most parallel to the tracked direction.
    // If the updated Hessian has lost it, recompute the exact Hessian once.
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(h_internal);
    detail::ModeChoice choice = detail::best_overlap_mode(tracked, basis * es.eigenvectors());
    if (choice.overlap < o.mode_overlap_threshold && r.n_recomputes < o.max_hessian_recomputes) {
      h = detail::exact_hessian(model, r.geometry, o, r);
      r.n_recomputes += 1;
      h_internal = basis.transpose() * h * basis;
      es.compute(h_internal);
      choice = detail::best_overlap_mode(tracked, basis * es.eigenvectors());
    }
    r.mode_overlaps.push_back(choice.overlap);

    const PrfoStep p = rs_prfo_step(h_internal, g_internal, trust, choice.index);
    const Eigen::VectorXd mode_full = basis * p.eigenvectors.col(choice.index);
    // Keep the tracked vector's sign continuous.
    tracked = (tracked.size() != 0 && tracked.dot(mode_full) < 0.0) ? Eigen::VectorXd(-mode_full)
                                                                      : mode_full;
    const Eigen::VectorXd step = basis * p.step;
    Geometry next = r.geometry;
    next.coords += step;
    Evaluation e = detail::evaluate_counted(model, next, r);
    r.step_norms.push_back(step.norm());
    h = bofill_update(h, step, e.gradient - r.evaluation.gradient);
    const double actual = e.energy.value() - r.evaluation.energy.value();
    trust = detail::update_trust(trust, detail::trust_ratio(actual, p.predicted_change),
                                 step.norm(), o);
    r.geometry = std::move(next);
    r.evaluation = std::move(e);
    r.energies.push_back(r.evaluation.energy.value());
  }
  r.converged = gradient_converged(internal_gradient(r.geometry, r.evaluation.gradient), o);
  r.hessian = std::move(h);
  r.followed_mode = tracked.size() != 0 ? Eigen::VectorXd(tracked.normalized()) : tracked;
  return r;
}

}  // namespace saddle::search
