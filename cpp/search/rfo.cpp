#include "search/rfo.hpp"

#include <cmath>
#include <stdexcept>

#include "hess/vibrations.hpp"

namespace saddle::search {

namespace {

constexpr int kBisectionIterations = 60;
constexpr double kTinyComponent = 1e-14;

// Lowest-eigenvalue RFO step with scaling alpha (Besalu and Bofill 1998).
Eigen::VectorXd rfo_min_component(const Eigen::VectorXd& b, const Eigen::VectorXd& f, double alpha) {
  const Eigen::Index n = b.size();
  if (n == 0) return Eigen::VectorXd(0);
  Eigen::MatrixXd m = Eigen::MatrixXd::Zero(n + 1, n + 1);
  const double sa = std::sqrt(alpha);
  m.topLeftCorner(n, n) = (b / alpha).asDiagonal();
  m.topRightCorner(n, 1) = f / sa;
  m.bottomLeftCorner(1, n) = (f / sa).transpose();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(m);
  const Eigen::VectorXd v = es.eigenvectors().col(0);
  const double t = v[n];
  if (std::abs(t) < kTinyComponent) {
    // Degenerate augmented problem; fall back to a Newton-like step on |b|.
    return -f.cwiseQuotient(b.cwiseAbs().cwiseMax(1e-8));
  }
  return v.head(n) / (t * sa);
}

}  // namespace

double rms(const Eigen::VectorXd& v) {
  return v.size() == 0 ? 0.0 : std::sqrt(v.squaredNorm() / static_cast<double>(v.size()));
}

Eigen::MatrixXd bfgs_update(const Eigen::MatrixXd& h, const Eigen::VectorXd& s, const Eigen::VectorXd& y) {
  const double ys = y.dot(s);
  if (!(ys > 1e-12 * s.norm() * y.norm())) return h;
  const Eigen::VectorXd hs = h * s;
  const double shs = s.dot(hs);
  if (!(shs > 0.0)) return h;
  return h + (y * y.transpose()) / ys - (hs * hs.transpose()) / shs;
}

Eigen::MatrixXd internal_basis(const Geometry& geometry) {
  const Eigen::Index n = geometry.dimension();
  const auto n_atoms = static_cast<Eigen::Index>(geometry.atomic_numbers.size());
  if (n_atoms < 2 || 3 * n_atoms != n) return Eigen::MatrixXd::Identity(n, n);
  const Eigen::MatrixXd ext = hess::external_basis(geometry.coords, Eigen::VectorXd::Ones(n_atoms));
  Eigen::HouseholderQR<Eigen::MatrixXd> qr(ext);
  const Eigen::MatrixXd q = qr.householderQ() * Eigen::MatrixXd::Identity(n, n);
  return q.rightCols(n - ext.cols());
}

RfoStep rs_rfo_minimize_step(const Eigen::MatrixXd& hessian, const Eigen::VectorXd& gradient, double trust) {
  if (!(trust > 0.0)) throw std::invalid_argument("trust radius must be positive");
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(0.5 * (hessian + hessian.transpose()));
  const Eigen::VectorXd b = es.eigenvalues();
  const Eigen::MatrixXd v = es.eigenvectors();
  const Eigen::VectorXd f = v.transpose() * gradient;

  auto step_at = [&](double alpha) { return Eigen::VectorXd(v * rfo_min_component(b, f, alpha)); };

  double alpha = 1.0;
  Eigen::VectorXd s = step_at(alpha);
  if (s.norm() > trust) {
    double lo = 1.0;
    double hi = 2.0;
    while (step_at(hi).norm() > trust && hi < 1e12) hi *= 4.0;
    for (int i = 0; i < kBisectionIterations; ++i) {
      const double mid = 0.5 * (lo + hi);
      (step_at(mid).norm() > trust ? lo : hi) = mid;
    }
    alpha = hi;
    s = step_at(alpha);
    if (s.norm() > trust) s *= trust / s.norm();  // guarantee the bound
  }
  const double predicted = gradient.dot(s) + 0.5 * s.dot(hessian * s);
  return RfoStep{s, predicted, alpha};
}

}  // namespace saddle::search
