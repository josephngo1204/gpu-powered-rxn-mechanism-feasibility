#pragma once

#include <Eigen/Dense>

namespace saddle::hess {

struct VibrationalAnalysis {
  // Vibrational frequencies (cm^-1), ascending. Imaginary modes are negative.
  Eigen::VectorXd frequencies_cm;
  // Normal modes as orthonormal columns in mass-weighted coordinates.
  Eigen::MatrixXd modes_mass_weighted;
  // Same modes as Cartesian displacements, each column normalized.
  Eigen::MatrixXd modes_cartesian;
  int n_external;                   // 6, or 5 for a linear geometry
  bool linear;
  double max_external_residual_cm;  // largest |frequency| inside the TR space
  double translation_sum_rule;      // max |sum_j H[(i,a),(j,b)]|, Eh/bohr^2

  [[nodiscard]] int n_imaginary(double threshold_cm) const {
    int n = 0;
    for (Eigen::Index i = 0; i < frequencies_cm.size(); ++i) {
      if (frequencies_cm[i] < -threshold_cm) ++n;
    }
    return n;
  }
};

// Harmonic analysis of a Cartesian Hessian (Eh/bohr^2) at coords (bohr) with
// masses in amu. Translations and rotations are removed by working in the
// orthogonal complement of the mass-weighted TR space (spec §6 "Vibrational
// projection"). The residual TR frequencies and sum rule are reported rather
// than hidden.
[[nodiscard]] VibrationalAnalysis analyze_vibrations(const Eigen::MatrixXd& hessian,
                                                     const Eigen::VectorXd& coords,
                                                     const Eigen::VectorXd& masses_amu);

// Mass-weighted orthonormal basis of translations and rotations (3N x 5 or 6).
[[nodiscard]] Eigen::MatrixXd external_basis(const Eigen::VectorXd& coords,
                                             const Eigen::VectorXd& masses_amu);

// Signed frequency in cm^-1 of a mass-weighted eigenvalue (Eh / (bohr^2 amu)).
[[nodiscard]] double eigenvalue_to_wavenumber(double eigenvalue);

}  // namespace saddle::hess
