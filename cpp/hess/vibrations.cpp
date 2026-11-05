#include "hess/vibrations.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#include "linalg/constants.hpp"
#include "linalg/view.hpp"

namespace saddle::hess {

namespace {

constexpr double kRankTolerance = 1e-8;

void validate(const Eigen::MatrixXd& hessian, const Eigen::VectorXd& coords,
              const Eigen::VectorXd& masses) {
  const Eigen::Index n3 = coords.size();
  if (n3 % 3 != 0 || masses.size() * 3 != n3) {
    throw ShapeError("vibrations: coords length " + std::to_string(n3) + " does not match " +
                     std::to_string(masses.size()) + " masses");
  }
  if (hessian.rows() != n3 || hessian.cols() != n3) {
    throw ShapeError("vibrations: Hessian is " + std::to_string(hessian.rows()) + "x" +
                     std::to_string(hessian.cols()) + ", expected " + std::to_string(n3));
  }
  if ((masses.array() <= 0.0).any()) throw std::invalid_argument("vibrations: masses must be positive");
}

}  // namespace

double eigenvalue_to_wavenumber(double eigenvalue) {
  const double omega = std::sqrt(std::abs(eigenvalue) / constants::kAmuToElectronMass) *
                       constants::kHartreeToWavenumber;
  return eigenvalue < 0.0 ? -omega : omega;
}

Eigen::VectorXd inverse_sqrt_masses(const Eigen::VectorXd& masses_amu) {
  Eigen::VectorXd out(3 * masses_amu.size());
  for (Eigen::Index i = 0; i < masses_amu.size(); ++i) {
    out.segment<3>(3 * i).setConstant(1.0 / std::sqrt(masses_amu[i]));
  }
  return out;
}

Eigen::MatrixXd orthogonal_complement(const Eigen::MatrixXd& basis) {
  const Eigen::Index n = basis.rows();
  const Eigen::HouseholderQR<Eigen::MatrixXd> qr(basis);
  const Eigen::MatrixXd q = qr.householderQ() * Eigen::MatrixXd::Identity(n, n);
  return q.rightCols(n - basis.cols());
}

Eigen::MatrixXd external_basis(const Eigen::VectorXd& coords, const Eigen::VectorXd& masses) {
  const Eigen::Index n = masses.size();
  const double total = masses.sum();
  Eigen::Vector3d com = Eigen::Vector3d::Zero();
  for (Eigen::Index i = 0; i < n; ++i) com += masses[i] * coords.segment<3>(3 * i);
  com /= total;

  Eigen::MatrixXd raw = Eigen::MatrixXd::Zero(3 * n, 6);
  for (Eigen::Index i = 0; i < n; ++i) {
    const double s = std::sqrt(masses[i]);
    const Eigen::Vector3d r = coords.segment<3>(3 * i) - com;
    for (Eigen::Index a = 0; a < 3; ++a) {
      raw(3 * i + a, a) = s;
      const Eigen::Vector3d axis = Eigen::Vector3d::Unit(a);
      raw.block<3, 1>(3 * i, 3 + a) = s * axis.cross(r);
    }
  }
  // Orthonormalize and keep the numerical rank (5 for linear geometries).
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(raw, Eigen::ComputeThinU);
  const auto& sv = svd.singularValues();
  Eigen::Index rank = 0;
  for (Eigen::Index k = 0; k < sv.size(); ++k) {
    if (sv[k] > kRankTolerance * sv[0]) ++rank;
  }
  return svd.matrixU().leftCols(rank);
}

VibrationalAnalysis analyze_vibrations(const Eigen::MatrixXd& hessian, const Eigen::VectorXd& coords,
                                       const Eigen::VectorXd& masses) {
  validate(hessian, coords, masses);
  const Eigen::Index n3 = coords.size();
  const Eigen::Index n_atoms = masses.size();

  const Eigen::VectorXd inv_sqrt_m = inverse_sqrt_masses(masses);
  const Eigen::MatrixXd hmw = inv_sqrt_m.asDiagonal() * hessian * inv_sqrt_m.asDiagonal();

  // Translation sum rule on the Cartesian Hessian.
  double sum_rule = 0.0;
  for (Eigen::Index row = 0; row < n3; ++row) {
    for (Eigen::Index b = 0; b < 3; ++b) {
      double s = 0.0;
      for (Eigen::Index j = 0; j < n_atoms; ++j) s += hessian(row, 3 * j + b);
      sum_rule = std::max(sum_rule, std::abs(s));
    }
  }

  const Eigen::MatrixXd ext = external_basis(coords, masses);
  const Eigen::Index n_ext = ext.cols();

  double max_ext = 0.0;
  if (n_ext > 0) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> ext_solver(ext.transpose() * hmw * ext);
    for (Eigen::Index k = 0; k < n_ext; ++k) {
      max_ext = std::max(max_ext, std::abs(eigenvalue_to_wavenumber(ext_solver.eigenvalues()[k])));
    }
  }

  const Eigen::MatrixXd internal = orthogonal_complement(ext);

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(internal.transpose() * hmw * internal);
  const Eigen::Index n_vib = n3 - n_ext;

  VibrationalAnalysis out;
  out.frequencies_cm.resize(n_vib);
  for (Eigen::Index k = 0; k < n_vib; ++k) out.frequencies_cm[k] = eigenvalue_to_wavenumber(solver.eigenvalues()[k]);
  out.modes_mass_weighted = internal * solver.eigenvectors();
  out.modes_cartesian = inv_sqrt_m.asDiagonal() * out.modes_mass_weighted;
  for (Eigen::Index k = 0; k < n_vib; ++k) out.modes_cartesian.col(k).normalize();
  out.n_external = static_cast<int>(n_ext);
  out.linear = (n_ext == 5);
  out.max_external_residual_cm = max_ext;
  out.translation_sum_rule = sum_rule;
  return out;
}

}  // namespace saddle::hess
