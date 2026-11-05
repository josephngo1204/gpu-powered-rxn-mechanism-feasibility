#pragma once

#include <Eigen/Dense>
#include <stdexcept>
#include <vector>

#include "core/model.hpp"
#include "linalg/view.hpp"

namespace saddle::hess {

enum class Stencil { Central, FivePoint };

struct DirectionalResult {
  Eigen::VectorXd hessian_vector;  // H v for unit v
  double curvature;                // v^T H v
  int n_gradients;
};

// H v by differences of gradients along unit direction v.
//   Central:   [g(x+hv) - g(x-hv)] / 2h                         O(h^2), 2 gradients
//   FivePoint: [-g(x+2hv) + 8g(x+hv) - 8g(x-hv) + g(x-2hv)] / 12h  O(h^4), 4 gradients
// FivePoint equals Richardson extrapolation of central differences at h and 2h.
template <ElectronicModel Model>
[[nodiscard]] DirectionalResult directional_curvature(Model& model, const Geometry& reference,
                                                      const Eigen::VectorXd& direction, double step,
                                                      Stencil stencil) {
  if (direction.size() != reference.dimension()) {
    throw ShapeError("direction does not match geometry dimension");
  }
  const double norm = direction.norm();
  if (!(norm > 0.0)) throw std::invalid_argument("direction must be nonzero");
  if (!(step > 0.0)) throw std::invalid_argument("step must be positive");
  const Eigen::VectorXd v = direction / norm;

  auto grad_at = [&](double t) {
    Geometry g = reference;
    g.coords += t * v;
    return model.evaluate(g, WarmStart{}).gradient;
  };

  Eigen::VectorXd hv;
  int n = 0;
  if (stencil == Stencil::Central) {
    hv = (grad_at(step) - grad_at(-step)) / (2.0 * step);
    n = 2;
  } else {
    hv = (-grad_at(2.0 * step) + 8.0 * grad_at(step) - 8.0 * grad_at(-step) + grad_at(-2.0 * step)) /
         (12.0 * step);
    n = 4;
  }
  return DirectionalResult{hv, v.dot(hv), n};
}

struct RefinedMode {
  double central;                // v^T H v from central differences
  double refined;                // five-point (Richardson) estimate
  double richardson_correction;  // refined - central
  int n_gradients;
};

// Mode-level Richardson for flagged modes only (spec §6). Columns of `modes`
// are Cartesian directions; each is normalized.
template <ElectronicModel Model>
[[nodiscard]] std::vector<RefinedMode> refine_modes(Model& model, const Geometry& reference,
                                                    const Eigen::MatrixXd& modes, double step) {
  if (modes.rows() != reference.dimension()) throw ShapeError("modes do not match geometry dimension");
  if (!(step > 0.0)) throw std::invalid_argument("step must be positive");
  std::vector<RefinedMode> out;
  out.reserve(static_cast<std::size_t>(modes.cols()));
  for (Eigen::Index k = 0; k < modes.cols(); ++k) {
    const double norm = modes.col(k).norm();
    if (!(norm > 0.0)) throw std::invalid_argument("mode columns must be nonzero");
    const Eigen::VectorXd v = modes.col(k) / norm;
    auto grad_at = [&](double t) {
      Geometry g = reference;
      g.coords += t * v;
      return model.evaluate(g, WarmStart{}).gradient;
    };
    // Four gradients per mode; the central estimate reuses the +-h points.
    const Eigen::VectorXd gp1 = grad_at(step), gm1 = grad_at(-step);
    const Eigen::VectorXd gp2 = grad_at(2.0 * step), gm2 = grad_at(-2.0 * step);
    const double c = v.dot(gp1 - gm1) / (2.0 * step);
    const double r = v.dot(-gp2 + 8.0 * gp1 - 8.0 * gm1 + gm2) / (12.0 * step);
    out.push_back({c, r, r - c, 4});
  }
  return out;
}

}  // namespace saddle::hess
