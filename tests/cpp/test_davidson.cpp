#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>

#include "hess/davidson.hpp"
#include "hess/fd_hessian.hpp"
#include "hess/partial_screen.hpp"
#include "hess/vibrations.hpp"
#include "surfaces/surfaces.hpp"

using namespace saddle;
using namespace saddle::hess;
using saddle::surfaces::Spring;
using saddle::surfaces::Springs;

namespace {

Eigen::MatrixXd random_symmetric(Eigen::Index n, unsigned seed) {
  std::srand(seed);
  Eigen::MatrixXd a = Eigen::MatrixXd::Random(n, n);
  return 0.5 * (a + a.transpose()) + Eigen::VectorXd::LinSpaced(n, -2.0, 10.0).asDiagonal().toDenseMatrix();
}

struct Chain {
  Geometry geometry;
  Springs model;
  Eigen::VectorXd masses;
};

Chain make_chain(int n, double k01) {
  Geometry g;
  g.coords.resize(3 * n);
  g.atomic_numbers.assign(static_cast<std::size_t>(n), 6);
  for (int i = 0; i < n; ++i) {
    g.coords.segment<3>(3 * i) << 1.4 * i, (i % 2 == 0 ? 0.0 : 1.2), 0.15 * std::sin(i);
  }
  auto dist = [&](int i, int j) { return (g.coords.segment<3>(3 * i) - g.coords.segment<3>(3 * j)).norm(); };
  std::vector<Spring> springs;
  for (int i = 0; i + 1 < n; ++i) springs.push_back({i, i + 1, i == 0 ? k01 : 0.4, dist(i, i + 1)});
  for (int i = 0; i + 2 < n; ++i) springs.push_back({i, i + 2, 0.1, dist(i, i + 2)});
  for (int i = 0; i + 3 < n; ++i) springs.push_back({i, i + 3, 0.05, dist(i, i + 3)});
  return {g, Springs{springs}, Eigen::VectorXd::Constant(n, 12.0)};
}

}

TEST(Davidson, FindsLowestEigenpairsOfDenseMatrix) {
  const Eigen::MatrixXd a = random_symmetric(60, 1);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> exact(a);
  DavidsonOptions opts;
  opts.n_roots = 3;
  opts.tolerance = 1e-8;
  opts.max_matvecs = 200;
  const auto result = davidson([&](const Eigen::VectorXd& v) { return Eigen::VectorXd(a * v); }, 60,
                               opts, a.diagonal());
  ASSERT_TRUE(result.converged);
  for (int k = 0; k < 3; ++k) EXPECT_NEAR(result.eigenvalues[k], exact.eigenvalues()[k], 1e-7);
  EXPECT_NEAR(std::abs(result.eigenvectors.col(0).dot(exact.eigenvectors().col(0))), 1.0, 1e-6);
}

TEST(Davidson, WorksWithoutPreconditioner) {
  const Eigen::MatrixXd a = random_symmetric(30, 2);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> exact(a);
  DavidsonOptions opts;
  opts.n_roots = 1;
  opts.tolerance = 1e-7;
  opts.max_matvecs = 100;
  const auto result = davidson([&](const Eigen::VectorXd& v) { return Eigen::VectorXd(a * v); }, 30, opts);
  ASSERT_TRUE(result.converged);
  EXPECT_NEAR(result.eigenvalues[0], exact.eigenvalues()[0], 1e-6);
}

TEST(Davidson, ReportsNonConvergenceInsteadOfPretending) {
  const Eigen::MatrixXd a = random_symmetric(40, 3);
  DavidsonOptions opts;
  opts.n_roots = 2;
  opts.tolerance = 1e-12;
  opts.max_matvecs = 4;
  const auto result = davidson([&](const Eigen::VectorXd& v) { return Eigen::VectorXd(a * v); }, 40, opts);
  EXPECT_FALSE(result.converged);
  EXPECT_LE(result.n_matvecs, 4);
}

TEST(Davidson, GoodInitialGuessNeedsFewerProducts) {
  const Eigen::MatrixXd a = random_symmetric(50, 4);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> exact(a);
  DavidsonOptions opts;
  opts.n_roots = 1;
  opts.tolerance = 1e-6;
  opts.max_matvecs = 100;
  auto op = [&](const Eigen::VectorXd& v) { return Eigen::VectorXd(a * v); };
  Eigen::MatrixXd guess = exact.eigenvectors().col(0) + 1e-3 * Eigen::VectorXd::Ones(50);
  const auto with_guess = davidson(op, 50, opts, {}, guess);
  const auto without = davidson(op, 50, opts);
  ASSERT_TRUE(with_guess.converged);
  ASSERT_TRUE(without.converged);
  EXPECT_LT(with_guess.n_matvecs, without.n_matvecs);
  EXPECT_NEAR(with_guess.eigenvalues[0], exact.eigenvalues()[0], 1e-8);
}

TEST(PartialScreen, LowestModeMatchesFullAnalysis) {
  auto chain = make_chain(8, -0.2);
  const auto full = analyze_vibrations(estimate_hessian(chain.model, chain.geometry, 1e-3).hessian,
                                       chain.geometry.coords, chain.masses);
  ScreenOptions opts;
  opts.n_roots = 2;
  opts.step = 1e-3;
  opts.tolerance_cm = 1.0;
  const auto screen = partial_hessian_screen(chain.model, chain.geometry, chain.masses, opts);
  ASSERT_TRUE(screen.converged);
  EXPECT_NEAR(screen.frequencies_cm[0], full.frequencies_cm[0], 1.0);
  EXPECT_NEAR(screen.frequencies_cm[1], full.frequencies_cm[1], 1.0);
  EXPECT_LT(screen.frequencies_cm[0], 0.0);
  EXPECT_EQ(screen.n_gradients, 2 * screen.n_matvecs);
}

TEST(PartialScreen, TwentyAtomCostIsWithinFifteenPercentOfFullHessian) {
  auto chain = make_chain(20, -0.2);
  ScreenOptions opts;
  opts.n_roots = 1;
  opts.step = 1e-3;
  opts.tolerance_cm = 5.0;
  Eigen::VectorXd guess = Eigen::VectorXd::Zero(60);
  const Eigen::Vector3d bond = (chain.geometry.coords.segment<3>(3) - chain.geometry.coords.segment<3>(0)).normalized();
  guess.segment<3>(0) = -bond;
  guess.segment<3>(3) = bond;
  const auto screen = partial_hessian_screen(chain.model, chain.geometry, chain.masses, opts, guess);
  ASSERT_TRUE(screen.converged);
  EXPECT_LT(screen.frequencies_cm[0], 0.0);
  const int full_cost = 2 * 60;
  EXPECT_LE(screen.n_gradients, static_cast<int>(0.15 * full_cost));
}
