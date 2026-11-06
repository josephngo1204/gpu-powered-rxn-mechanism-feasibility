#include "hess/fd_hessian.hpp"

#include <stdexcept>
#include <string>

namespace saddle::hess {

std::vector<StencilPoint> central_stencil(const Geometry& reference, double step) {
  if (!(step > 0.0)) throw std::invalid_argument("stencil step must be positive");
  const Eigen::Index n = reference.dimension();
  std::vector<StencilPoint> points;
  points.reserve(static_cast<std::size_t>(2 * n));
  for (Eigen::Index i = 0; i < n; ++i) {
    for (int sign : {+1, -1}) {
      Geometry displaced = reference;
      displaced.coords[i] += sign * step;
      points.push_back({i, sign, std::move(displaced)});
    }
  }
  return points;
}

double relative_asymmetry(const Eigen::MatrixXd& matrix) {
  const double norm = matrix.norm();
  if (norm == 0.0) return 0.0;
  return (matrix - matrix.transpose()).norm() / norm;
}

HessianResult assemble_central(const std::vector<Eigen::VectorXd>& gradients, double step) {
  if (!(step > 0.0)) throw std::invalid_argument("stencil step must be positive");
  if (gradients.empty() || gradients.size() % 2 != 0) {
    throw ShapeError("central stencil needs an even, nonzero number of gradients");
  }
  const auto n = static_cast<Eigen::Index>(gradients.size() / 2);
  Eigen::MatrixXd raw(n, n);
  for (Eigen::Index i = 0; i < n; ++i) {
    const auto& plus = gradients[static_cast<std::size_t>(2 * i)];
    const auto& minus = gradients[static_cast<std::size_t>(2 * i + 1)];
    if (plus.size() != n || minus.size() != n) {
      throw ShapeError("gradient length " + std::to_string(plus.size()) +
                       " does not match stencil dimension " + std::to_string(n));
    }
    raw.col(i) = (plus - minus) / (2.0 * step);
  }
  Eigen::MatrixXd symmetric = 0.5 * (raw + raw.transpose());
  const double asym = relative_asymmetry(raw);
  return HessianResult{std::move(raw), std::move(symmetric), asym, step,
                       static_cast<int>(gradients.size())};
}

}
