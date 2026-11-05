#pragma once

#include <Eigen/Dense>
#include <vector>

#include "core/model.hpp"
#include "linalg/view.hpp"

namespace saddle::hess {

// One displaced geometry of a finite-difference stencil.
struct StencilPoint {
  Eigen::Index coordinate;
  int sign;  // +1 or -1
  Geometry geometry;
};

struct HessianResult {
  Eigen::MatrixXd raw;      // H[:, i] = (g(x + h e_i) - g(x - h e_i)) / 2h
  Eigen::MatrixXd hessian;  // (raw + raw^T) / 2
  double asymmetry;         // ||raw - raw^T||_F / ||raw||_F
  double step;
  int n_gradients;
};

// Central-difference stencil: points ordered (0,+), (0,-), (1,+), (1,-), ...
// Each pair is matched: same settings, opposite displacement (spec §6).
[[nodiscard]] std::vector<StencilPoint> central_stencil(const Geometry& reference, double step);

// Assemble a Hessian from gradients evaluated at central_stencil points, in order.
[[nodiscard]] HessianResult assemble_central(const std::vector<Eigen::VectorXd>& gradients,
                                             double step);

[[nodiscard]] double relative_asymmetry(const Eigen::MatrixXd& matrix);

// Serial convenience driver. Production runs execute stencil points through
// the restartable task store instead.
template <ElectronicModel Model>
[[nodiscard]] HessianResult estimate_hessian(Model& model, const Geometry& reference,
                                             double step) {
  const auto points = central_stencil(reference, step);
  std::vector<Eigen::VectorXd> gradients;
  gradients.reserve(points.size());
  for (const auto& point : points) {
    gradients.push_back(model.evaluate(point.geometry, WarmStart{}).gradient);
  }
  return assemble_central(gradients, step);
}

}  // namespace saddle::hess
