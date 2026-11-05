#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/model.hpp"
#include "linalg/view.hpp"
#include "search/optimizer.hpp"

namespace saddle::search {

struct IrcOptions {
  double step = 0.05;            // arc length per step, mass-weighted (bohr amu^1/2)
  int max_steps = 300;
  double switch_gradient = 1e-3;  // hand over to the minimizer below this |g| (Eh/bohr)
  OptimizerOptions endpoint;     // endpoint optimization settings
};

struct IrcBranch {
  std::vector<Geometry> path;  // starts at the TS
  std::vector<double> energies;
  OptimizationResult endpoint;
  std::string stop_reason;
};

struct IrcResult {
  IrcBranch forward;  // along +transition_mode
  IrcBranch reverse;
  Eigen::VectorXd transition_mode;  // Cartesian unit vector
  double transition_eigenvalue;     // mass-weighted, negative
};

namespace detail {

template <ElectronicModel Model>
IrcBranch irc_branch(Model& model, const Geometry& ts, double ts_energy, const Eigen::VectorXd& inv_sqrt_m,
                     const Eigen::VectorXd& mode_mw, double sign, const IrcOptions& o) {
  IrcBranch b;
  b.path.push_back(ts);
  b.energies.push_back(ts_energy);
  // Initial displacement along the transition mode.
  Geometry x = ts;
  x.coords += inv_sqrt_m.cwiseProduct(sign * o.step * mode_mw);
  Evaluation e = model.evaluate(x, WarmStart{});
  b.stop_reason = "max_steps";
  auto descent = [&](const Eigen::VectorXd& g) {
    const Eigen::VectorXd gq = inv_sqrt_m.cwiseProduct(g);  // mass-weighted gradient
    return Eigen::VectorXd(-gq / gq.norm());
  };
  for (int k = 0; k < o.max_steps; ++k) {
    if (e.energy.value() >= b.energies.back()) {
      b.stop_reason = "energy_rise";
      break;
    }
    b.path.push_back(x);
    b.energies.push_back(e.energy.value());
    if (e.gradient.norm() < o.switch_gradient) {
      b.stop_reason = "small_gradient";
      break;
    }
    // Heun predictor-corrector in mass-weighted coordinates.
    const Eigen::VectorXd d0 = descent(e.gradient);
    Geometry pred = x;
    pred.coords += inv_sqrt_m.cwiseProduct(o.step * d0);
    const Evaluation ep = model.evaluate(pred, WarmStart{});
    Eigen::VectorXd d = d0 + descent(ep.gradient);
    if (d.norm() < 1e-12) d = d0;  // passed a minimum between points
    Geometry next = x;
    next.coords += inv_sqrt_m.cwiseProduct(o.step * d.normalized());
    x = std::move(next);
    e = model.evaluate(x, WarmStart{});
  }
  b.endpoint = minimize(model, b.path.back(), o.endpoint);
  return b;
}

}  // namespace detail

// Intrinsic reaction coordinate from a first-order saddle in both directions,
// followed by endpoint optimization. `masses_amu` may be empty for model
// surfaces (unit masses).
template <ElectronicModel Model>
[[nodiscard]] IrcResult irc(Model& model, const Geometry& ts, const Eigen::MatrixXd& hessian,
                            const Eigen::VectorXd& masses_amu, const IrcOptions& o) {
  const Eigen::Index n = ts.dimension();
  if (hessian.rows() != n || hessian.cols() != n) throw ShapeError("IRC: Hessian does not match geometry");
  Eigen::VectorXd inv_sqrt_m = Eigen::VectorXd::Ones(n);
  if (masses_amu.size() != 0) {
    if (3 * masses_amu.size() != n) throw ShapeError("IRC: masses do not match geometry");
    for (Eigen::Index i = 0; i < masses_amu.size(); ++i) inv_sqrt_m.segment<3>(3 * i).setConstant(1.0 / std::sqrt(masses_amu[i]));
  }
  const Eigen::MatrixXd basis = internal_basis(ts);
  const Eigen::MatrixXd hmw = inv_sqrt_m.asDiagonal() * hessian * inv_sqrt_m.asDiagonal();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(basis.transpose() * hmw * basis);
  if (!(es.eigenvalues()[0] < 0.0) || (es.eigenvalues().size() > 1 && es.eigenvalues()[1] < 0.0)) {
    throw std::invalid_argument("IRC needs a first-order saddle (exactly one negative eigenvalue)");
  }
  Eigen::VectorXd mode_mw = basis * es.eigenvectors().col(0);
  // Deterministic sign: largest-magnitude Cartesian component is positive.
  Eigen::VectorXd mode_cart = inv_sqrt_m.cwiseProduct(mode_mw);
  Eigen::Index imax = 0;
  mode_cart.cwiseAbs().maxCoeff(&imax);
  if (mode_cart[imax] < 0.0) {
    mode_mw = -mode_mw;
    mode_cart = -mode_cart;
  }
  const double e0 = model.evaluate(ts, WarmStart{}).energy.value();
  IrcResult r;
  r.transition_mode = mode_cart.normalized();
  r.transition_eigenvalue = es.eigenvalues()[0];
  r.forward = detail::irc_branch(model, ts, e0, inv_sqrt_m, mode_mw, +1.0, o);
  r.reverse = detail::irc_branch(model, ts, e0, inv_sqrt_m, mode_mw, -1.0, o);
  return r;
}

}  // namespace saddle::search
