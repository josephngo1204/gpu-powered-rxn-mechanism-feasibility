#pragma once

#include <Eigen/Dense>

#include "core/model.hpp"

namespace saddle::search {

[[nodiscard]] double rms(const Eigen::VectorXd& v);

// BFGS update; skipped (returns h) when the curvature condition y.s > 0 fails.
[[nodiscard]] Eigen::MatrixXd bfgs_update(const Eigen::MatrixXd& h, const Eigen::VectorXd& s,
                                          const Eigen::VectorXd& y);

// Orthonormal basis of the internal (non-translational, non-rotational)
// subspace. For geometries without atoms it is the identity.
[[nodiscard]] Eigen::MatrixXd internal_basis(const Geometry& geometry);

struct RfoStep {
  Eigen::VectorXd step;
  double predicted_change;  // g.s + 1/2 s.H.s
  double alpha;             // restricted-step scaling (1 = unrestricted)
};

// Restricted-step rational function optimization step for minimization:
// solves the augmented Hessian with scaling alpha >= 1 so |step| <= trust.
[[nodiscard]] RfoStep rs_rfo_minimize_step(const Eigen::MatrixXd& hessian,
                                           const Eigen::VectorXd& gradient, double trust);

}  // namespace saddle::search
