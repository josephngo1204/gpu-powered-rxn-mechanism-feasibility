#pragma once

#include <Eigen/Dense>

namespace saddle::hess {

struct VibrationalAnalysis {
  Eigen::VectorXd frequencies_cm;
  Eigen::MatrixXd modes_mass_weighted;
  Eigen::MatrixXd modes_cartesian;
  int n_external;
  bool linear;
  double max_external_residual_cm;
  double translation_sum_rule;

  [[nodiscard]] int n_imaginary(double threshold_cm) const {
    int n = 0;
    for (Eigen::Index i = 0; i < frequencies_cm.size(); ++i) {
      if (frequencies_cm[i] < -threshold_cm) ++n;
    }
    return n;
  }
};

[[nodiscard]] VibrationalAnalysis analyze_vibrations(const Eigen::MatrixXd& hessian,
                                                     const Eigen::VectorXd& coords,
                                                     const Eigen::VectorXd& masses_amu);

[[nodiscard]] Eigen::MatrixXd external_basis(const Eigen::VectorXd& coords,
                                             const Eigen::VectorXd& masses_amu);

[[nodiscard]] Eigen::VectorXd inverse_sqrt_masses(const Eigen::VectorXd& masses_amu);

[[nodiscard]] Eigen::MatrixXd orthogonal_complement(const Eigen::MatrixXd& basis);

[[nodiscard]] double eigenvalue_to_wavenumber(double eigenvalue);

}
