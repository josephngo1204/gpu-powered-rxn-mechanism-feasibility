#pragma once

#include <Eigen/Dense>
#include <functional>

namespace saddle::hess {

struct DavidsonOptions {
  int n_roots = 1;
  double tolerance = 1e-6;  // residual norm ||A x - theta x||
  int max_matvecs = 100;
  int max_subspace = 40;
};

struct DavidsonResult {
  Eigen::VectorXd eigenvalues;   // lowest n_roots, ascending
  Eigen::MatrixXd eigenvectors;  // unit columns
  Eigen::VectorXd residual_norms;
  int n_matvecs;
  bool converged;
};

using MatVec = std::function<Eigen::VectorXd(const Eigen::VectorXd&)>;

// Lowest eigenpairs of a symmetric operator known only through products.
// `diagonal` (optional) enables the Davidson preconditioner; `guess`
// (optional) supplies start vectors as columns. Returns converged = false
// when the matvec budget runs out, never a silently unconverged answer.
[[nodiscard]] DavidsonResult davidson(const MatVec& matvec, Eigen::Index dimension,
                                      const DavidsonOptions& options,
                                      const Eigen::VectorXd& diagonal = {},
                                      const Eigen::MatrixXd& guess = {});

}  // namespace saddle::hess
