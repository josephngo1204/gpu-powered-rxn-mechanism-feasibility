#pragma once

#include <Eigen/Dense>
#include <vector>

#include "core/model.hpp"
#include "linalg/view.hpp"

namespace saddle::hess {

struct StencilPoint {
  Eigen::Index coordinate;
  int sign;
  Geometry geometry;
};

struct HessianResult {
  Eigen::MatrixXd raw;
  Eigen::MatrixXd hessian;
  double asymmetry;
  double step;
  int n_gradients;
};

[[nodiscard]] std::vector<StencilPoint> central_stencil(const Geometry& reference, double step);

[[nodiscard]] HessianResult assemble_central(const std::vector<Eigen::VectorXd>& gradients,
                                             double step);

[[nodiscard]] double relative_asymmetry(const Eigen::MatrixXd& matrix);

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

}
