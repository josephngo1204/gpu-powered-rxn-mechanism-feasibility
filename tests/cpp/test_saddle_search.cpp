#include <gtest/gtest.h>

#include <Eigen/Dense>

#include "hess/fd_hessian.hpp"
#include "hess/vibrations.hpp"
#include "search/optimizer.hpp"
#include "surfaces/surfaces.hpp"

using namespace saddle;
using namespace saddle::search;
using saddle::surfaces::MullerBrown;
using saddle::surfaces::Spring;
using saddle::surfaces::Springs;

namespace {

int negative_eigenvalues(const Eigen::MatrixXd& h) {
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(h);
  return static_cast<int>((es.eigenvalues().array() < 0.0).count());
}

Springs triangle_ts() {
  return Springs{{Spring{0, 1, -0.05, 1.8}, Spring{0, 2, 0.5, 1.8}, Spring{1, 2, 0.2, 2.9}}};
}

Geometry distorted_triangle() {
  Geometry g{{8, 1, 1}, Eigen::VectorXd(9)};
  // Rest geometry: |01| = |02| = 1.8, |12| = 2.9; perturbed.
  const double half = 1.45, h = std::sqrt(1.8 * 1.8 - half * half);
  g.coords << 0.0, 0.0, 0.0, half + 0.08, h - 0.05, 0.02, -half, h + 0.04, -0.03;
  return g;
}

}  // namespace

TEST(Bofill, SatisfiesSecantConditionAndSymmetry) {
  Eigen::MatrixXd h(3, 3);
  h << 1.0, 0.2, 0.0, 0.2, -0.5, 0.1, 0.0, 0.1, 2.0;
  const Eigen::Vector3d s(0.1, -0.2, 0.05);
  const Eigen::Vector3d y(0.3, -0.1, 0.2);
  const Eigen::MatrixXd h1 = bofill_update(h, s, y);
  EXPECT_TRUE((h1 * s).isApprox(y, 1e-12));
  EXPECT_TRUE(h1.isApprox(h1.transpose(), 1e-14));
}

TEST(Bofill, CanCreateNegativeCurvature) {
  const Eigen::MatrixXd h = Eigen::MatrixXd::Identity(2, 2);
  const Eigen::Vector2d s(0.1, 0.0);
  const Eigen::Vector2d y(-0.1, 0.0);  // negative curvature along s
  EXPECT_EQ(negative_eigenvalues(bofill_update(h, s, y)), 1);
}

struct SaddleCase {
  double x0, y0, x, y, energy;
};

class MullerBrownSaddles : public ::testing::TestWithParam<SaddleCase> {};

TEST_P(MullerBrownSaddles, ConvergesToKnownFirstOrderSaddle) {
  const auto c = GetParam();
  MullerBrown mb;
  SaddleOptions opt;
  const auto r = saddle_search(mb, Geometry::cartesian({c.x0, c.y0}), opt);
  ASSERT_TRUE(r.converged) << "iterations " << r.iterations;
  EXPECT_NEAR(r.geometry.coords[0], c.x, 1e-3);
  EXPECT_NEAR(r.geometry.coords[1], c.y, 1e-3);
  EXPECT_NEAR(r.evaluation.energy.value(), c.energy, 1e-2);
  EXPECT_EQ(negative_eigenvalues(hess::estimate_hessian(mb, r.geometry, 1e-5).hessian), 1);
}

INSTANTIATE_TEST_SUITE_P(Known, MullerBrownSaddles,
                         ::testing::Values(SaddleCase{-0.75, 0.55, -0.822, 0.624, -40.665},
                                           SaddleCase{0.25, 0.25, 0.212, 0.293, -72.249}));

TEST(SaddleSearch, MolecularSaddleWithOneImaginaryMode) {
  auto model = triangle_ts();
  SaddleOptions opt;
  const auto r = saddle_search(model, distorted_triangle(), opt);
  ASSERT_TRUE(r.converged);
  const auto hess = hess::estimate_hessian(model, r.geometry, 1e-4);
  const auto vib = hess::analyze_vibrations(hess.hessian, r.geometry.coords,
                                            Eigen::Vector3d(16.0, 1.0, 1.0));
  EXPECT_EQ(vib.n_imaginary(1.0), 1);
}

TEST(SaddleSearch, FollowsRequestedDirection) {
  MullerBrown mb;
  SaddleOptions opt;
  opt.follow_direction = Eigen::Vector2d(1.0, -0.3);
  const auto r = saddle_search(mb, Geometry::cartesian({0.25, 0.25}), opt);
  ASSERT_TRUE(r.converged);
  EXPECT_NEAR(r.evaluation.energy.value(), -72.249, 1e-2);
}

TEST(SaddleSearch, OverlapThresholdTriggersHessianRecompute) {
  MullerBrown mb;
  SaddleOptions opt;
  opt.mode_overlap_threshold = 1.01;  // unattainable: recompute every iteration
  opt.max_hessian_recomputes = 3;
  const auto r = saddle_search(mb, Geometry::cartesian({0.25, 0.25}), opt);
  EXPECT_EQ(r.n_hessians, 1 + 3);
  EXPECT_EQ(r.n_recomputes, 3);
}

TEST(SaddleSearch, ReportsNonConvergence) {
  MullerBrown mb;
  SaddleOptions opt;
  opt.max_iterations = 1;
  const auto r = saddle_search(mb, Geometry::cartesian({-0.6, 0.4}), opt);
  EXPECT_FALSE(r.converged);
}

TEST(PrfoStep, RespectsTrustAndClimbsAlongChosenMode) {
  Eigen::MatrixXd h(2, 2);
  h << 1.0, 0.0, 0.0, 2.0;
  const Eigen::Vector2d g(0.1, 0.1);
  const auto step = rs_prfo_step(h, g, 0.05, 0);
  EXPECT_LE(step.step.norm(), 0.05 + 1e-12);
  EXPECT_GT(step.step[0], 0.0);  // uphill along mode 0 despite positive curvature
  EXPECT_LT(step.step[1], 0.0);  // downhill along mode 1
}
