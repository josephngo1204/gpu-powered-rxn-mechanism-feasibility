#pragma once

#include <Eigen/Dense>

#include "core/model.hpp"

namespace saddle::search {

[[nodiscard]] double rms(const Eigen::VectorXd& v);

[[nodiscard]] Eigen::MatrixXd bfgs_update(const Eigen::MatrixXd& h, const Eigen::VectorXd& s,
                                          const Eigen::VectorXd& y);

[[nodiscard]] Eigen::MatrixXd internal_basis(const Geometry& geometry);

[[nodiscard]] Eigen::VectorXd internal_gradient(const Geometry& geometry, const Eigen::VectorXd& gradient);

struct RfoStep {
  Eigen::VectorXd step;
  double predicted_change;
  double alpha;
};

[[nodiscard]] RfoStep rs_rfo_minimize_step(const Eigen::MatrixXd& hessian,
                                           const Eigen::VectorXd& gradient, double trust);

[[nodiscard]] Eigen::MatrixXd bofill_update(const Eigen::MatrixXd& h, const Eigen::VectorXd& s,
                                            const Eigen::VectorXd& y);

struct PrfoStep {
  Eigen::VectorXd step;
  double predicted_change;
  double alpha;
  Eigen::VectorXd eigenvalues;
  Eigen::MatrixXd eigenvectors;
};

[[nodiscard]] PrfoStep rs_prfo_step(const Eigen::MatrixXd& hessian, const Eigen::VectorXd& gradient,
                                    double trust, Eigen::Index mode);

}
