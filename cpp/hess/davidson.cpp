#include "hess/davidson.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "linalg/view.hpp"

namespace saddle::hess {

namespace {

constexpr double kMinNewNorm = 1e-10;

double orthogonalize(Eigen::VectorXd& v, const Eigen::MatrixXd& basis, Eigen::Index used) {
  for (int pass = 0; pass < 2; ++pass) {
    if (used > 0) v -= basis.leftCols(used) * (basis.leftCols(used).transpose() * v);
  }
  return v.norm();
}

}

DavidsonResult davidson(const MatVec& matvec, Eigen::Index n, const DavidsonOptions& opt,
                        const Eigen::VectorXd& diagonal, const Eigen::MatrixXd& guess) {
  if (n <= 0 || opt.n_roots < 1 || opt.n_roots > n) {
    throw std::invalid_argument("davidson: need 1 <= n_roots <= dimension");
  }
  if (diagonal.size() != 0 && diagonal.size() != n) throw ShapeError("davidson: diagonal size mismatch");
  if (guess.size() != 0 && guess.rows() != n) throw ShapeError("davidson: guess size mismatch");
  const Eigen::Index k = opt.n_roots;
  const Eigen::Index max_sub = std::min<Eigen::Index>(n, std::max<Eigen::Index>(opt.max_subspace, 2 * k + 1));

  Eigen::MatrixXd v(n, max_sub);
  Eigen::MatrixXd w(n, max_sub);
  Eigen::Index used = 0;
  int matvecs = 0;

  auto add = [&](Eigen::VectorXd t) {
    if (used >= max_sub || matvecs >= opt.max_matvecs) return false;
    const double norm = orthogonalize(t, v, used);
    if (norm < kMinNewNorm) return false;
    v.col(used) = t / norm;
    w.col(used) = matvec(v.col(used));
    ++matvecs;
    ++used;
    return true;
  };

  for (Eigen::Index c = 0; c < guess.cols(); ++c) add(guess.col(c));
  std::vector<Eigen::Index> order(static_cast<std::size_t>(n));
  for (Eigen::Index i = 0; i < n; ++i) order[static_cast<std::size_t>(i)] = i;
  if (diagonal.size() == n) {
    std::stable_sort(order.begin(), order.end(), [&](Eigen::Index a, Eigen::Index b) { return diagonal[a] < diagonal[b]; });
  }
  for (Eigen::Index i : order) {
    if (used >= k) break;
    add(Eigen::VectorXd::Unit(n, i));
  }

  DavidsonResult result;
  result.converged = false;
  while (true) {
    const Eigen::MatrixXd sub = v.leftCols(used).transpose() * w.leftCols(used);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(0.5 * (sub + sub.transpose()));
    const Eigen::Index roots = std::min(k, used);
    const Eigen::MatrixXd s = es.eigenvectors().leftCols(roots);
    const Eigen::VectorXd theta = es.eigenvalues().head(roots);
    const Eigen::MatrixXd x = v.leftCols(used) * s;
    const Eigen::MatrixXd r = w.leftCols(used) * s - x * theta.asDiagonal();

    result.eigenvalues = theta;
    result.eigenvectors = x;
    result.residual_norms = r.colwise().norm().transpose();
    result.n_matvecs = matvecs;
    result.converged = roots == k && (result.residual_norms.array() <= opt.tolerance).all();
    if (result.converged || matvecs >= opt.max_matvecs) return result;

    if (used + k > max_sub) {
      v.leftCols(roots) = x;
      w.leftCols(roots) = w.leftCols(used) * s;
      used = roots;
    }

    bool grew = false;
    for (Eigen::Index j = 0; j < roots; ++j) {
      if (result.residual_norms[j] <= opt.tolerance) continue;
      Eigen::VectorXd t = r.col(j);
      if (diagonal.size() == n) {
        for (Eigen::Index i = 0; i < n; ++i) {
          const double denom = theta[j] - diagonal[i];
          t[i] /= (std::abs(denom) > 1e-8 ? denom : 1e-8);
        }
      }
      grew = add(std::move(t)) || grew;
    }
    if (!grew) return result;
  }
}

}
