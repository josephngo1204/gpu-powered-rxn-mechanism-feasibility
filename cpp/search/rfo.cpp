#include "search/rfo.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

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

// Highest-eigenvalue RFO step along a single mode.
double rfo_max_component(double b, double f, double alpha) {
  const double sa = std::sqrt(alpha);
  Eigen::Matrix2d m;
  m << b / alpha, f / sa, f / sa, 0.0;
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(m);
  const Eigen::Vector2d v = es.eigenvectors().col(1);
  if (std::abs(v[1]) < kTinyComponent) return f / std::max(std::abs(b), 1e-8);
  return v[0] / (v[1] * sa);
}

// Shrink a step into the trust radius. `step_at(alpha)` must give a step whose
// length falls as the RFO scaling alpha >= 1 grows. Returns alpha = 1 when the
// unscaled step already fits; otherwise bisects for the smallest alpha that
// fits, and clips the final step in case bisection stops just outside.
template <class StepAt>
std::pair<double, Eigen::VectorXd> restrict_to_trust(const StepAt& step_at, double trust) {
  Eigen::VectorXd step = step_at(1.0);
  if (step.norm() <= trust) return {1.0, step};
  double lo = 1.0;
  double hi = 2.0;
  while (step_at(hi).norm() > trust && hi < 1e12) hi *= 4.0;
  for (int i = 0; i < kBisectionIterations; ++i) {
    const double mid = 0.5 * (lo + hi);
    (step_at(mid).norm() > trust ? lo : hi) = mid;
  }
  step = step_at(hi);
  if (step.norm() > trust) step *= trust / step.norm();
  return {hi, step};
}

// Quadratic-model energy change g.s + 1/2 s.H.s.
double predicted_change(const Eigen::VectorXd& gradient, const Eigen::MatrixXd& hessian,
                        const Eigen::VectorXd& step) {
  return gradient.dot(step) + 0.5 * step.dot(hessian * step);
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
  return hess::orthogonal_complement(
      hess::external_basis(geometry.coords, Eigen::VectorXd::Ones(n_atoms)));
}

Eigen::VectorXd internal_gradient(const Geometry& geometry, const Eigen::VectorXd& gradient) {
  const Eigen::MatrixXd b = internal_basis(geometry);
  return b * (b.transpose() * gradient);
}

RfoStep rs_rfo_minimize_step(const Eigen::MatrixXd& hessian, const Eigen::VectorXd& gradient, double trust) {
  if (!(trust > 0.0)) throw std::invalid_argument("trust radius must be positive");
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(0.5 * (hessian + hessian.transpose()));
  const Eigen::VectorXd b = es.eigenvalues();
  const Eigen::MatrixXd v = es.eigenvectors();
  const Eigen::VectorXd f = v.transpose() * gradient;

  auto step_at = [&](double alpha) { return Eigen::VectorXd(v * rfo_min_component(b, f, alpha)); };
  auto [alpha, s] = restrict_to_trust(step_at, trust);
  return RfoStep{s, predicted_change(gradient, hessian, s), alpha};
}

Eigen::MatrixXd bofill_update(const Eigen::MatrixXd& h, const Eigen::VectorXd& s, const Eigen::VectorXd& y) {
  const double ss = s.squaredNorm();
  if (!(ss > 0.0)) return h;
  const Eigen::VectorXd xi = y - h * s;
  const double xs = xi.dot(s);
  const double xx = xi.squaredNorm();
  if (xx == 0.0) return h;
  const double phi = xs * xs / (ss * xx);
  Eigen::MatrixXd sr1 = Eigen::MatrixXd::Zero(h.rows(), h.cols());
  if (std::abs(xs) > 1e-12 * std::sqrt(ss * xx)) sr1 = (xi * xi.transpose()) / xs;
  const Eigen::MatrixXd psb =
      (xi * s.transpose() + s * xi.transpose()) / ss - (xs / (ss * ss)) * (s * s.transpose());
  return h + phi * sr1 + (1.0 - phi) * psb;
}

PrfoStep rs_prfo_step(const Eigen::MatrixXd& hessian, const Eigen::VectorXd& gradient, double trust,
                      Eigen::Index mode) {
  if (!(trust > 0.0)) throw std::invalid_argument("trust radius must be positive");
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(0.5 * (hessian + hessian.transpose()));
  const Eigen::VectorXd b = es.eigenvalues();
  const Eigen::MatrixXd v = es.eigenvectors();
  const Eigen::Index n = b.size();
  if (mode < 0 || mode >= n) throw std::invalid_argument("P-RFO mode index out of range");
  const Eigen::VectorXd f = v.transpose() * gradient;

  // Split the eigenbasis: maximize along `mode`, minimize along all others.
  Eigen::VectorXd b_min(n - 1), f_min(n - 1);
  for (Eigen::Index i = 0, j = 0; i < n; ++i) {
    if (i == mode) continue;
    b_min[j] = b[i];
    f_min[j] = f[i];
    ++j;
  }
  auto step_at = [&](double alpha) {
    const Eigen::VectorXd s_min = rfo_min_component(b_min, f_min, alpha);
    Eigen::VectorXd s_eigen(n);
    for (Eigen::Index i = 0, j = 0; i < n; ++i) {
      s_eigen[i] = (i == mode) ? rfo_max_component(b[mode], f[mode], alpha) : s_min[j++];
    }
    return Eigen::VectorXd(v * s_eigen);
  };
  auto [alpha, s] = restrict_to_trust(step_at, trust);
  return PrfoStep{s, predicted_change(gradient, hessian, s), alpha, b, v};
}

}  // namespace saddle::search
