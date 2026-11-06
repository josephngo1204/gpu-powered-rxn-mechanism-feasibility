#pragma once

#include <Eigen/Dense>
#include <functional>

namespace saddle::hess {

struct DavidsonOptions {
  int n_roots = 1;
  double tolerance = 1e-6;
  int max_matvecs = 100;
  int max_subspace = 40;
};

struct DavidsonResult {
  Eigen::VectorXd eigenvalues;
  Eigen::MatrixXd eigenvectors;
  Eigen::VectorXd residual_norms;
  int n_matvecs;
  bool converged;
};

using MatVec = std::function<Eigen::VectorXd(const Eigen::VectorXd&)>;

[[nodiscard]] DavidsonResult davidson(const MatVec& matvec, Eigen::Index dimension,
                                      const DavidsonOptions& options,
                                      const Eigen::VectorXd& diagonal = {},
                                      const Eigen::MatrixXd& guess = {});

}
