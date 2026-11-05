#include <gtest/gtest.h>

#include <Eigen/Dense>

#include "search/optimizer.hpp"
#include "surfaces/surfaces.hpp"

using namespace saddle;
using namespace saddle::search;
using saddle::surfaces::MullerBrown;
using saddle::surfaces::Spring;
using saddle::surfaces::Springs;

struct MinimumCase {
  double x0, y0, x, y, energy;
};

class MullerBrownMinima : public ::testing::TestWithParam<MinimumCase> {};

TEST_P(MullerBrownMinima, ConvergesToKnownMinimum) {
  const auto c = GetParam();
  MullerBrown mb;
  OptimizerOptions opt;
  opt.initial_hessian = InitialHessian::Exact;
  const auto r = minimize(mb, Geometry::cartesian({c.x0, c.y0}), opt);
  ASSERT_TRUE(r.converged) << "iterations " << r.iterations;
  EXPECT_NEAR(r.geometry.coords[0], c.x, 1e-3);
  EXPECT_NEAR(r.geometry.coords[1], c.y, 1e-3);
  EXPECT_NEAR(r.evaluation.energy.value(), c.energy, 1e-2);
  EXPECT_LE(rms(r.evaluation.gradient), 1e-5);
  EXPECT_LE(r.evaluation.gradient.cwiseAbs().maxCoeff(), 1.5e-5);
}

INSTANTIATE_TEST_SUITE_P(Known, MullerBrownMinima,
                         ::testing::Values(MinimumCase{-0.7, 1.3, -0.558, 1.442, -146.700},
                                           MinimumCase{0.5, 0.1, 0.623, 0.028, -108.167},
                                           MinimumCase{-0.1, 0.5, -0.050, 0.467, -80.768}));

TEST(Minimize, IdentityStartHessianStillConvergesWithBfgs) {
  MullerBrown mb;
  OptimizerOptions opt;
  opt.initial_hessian = InitialHessian::Identity;
  opt.identity_scale = 500.0;
  opt.max_iterations = 300;
  const auto r = minimize(mb, Geometry::cartesian({0.4, 0.2}), opt);
  ASSERT_TRUE(r.converged);
  EXPECT_NEAR(r.evaluation.energy.value(), -108.167, 1e-2);
}

TEST(Minimize, ReportsNonConvergence) {
  MullerBrown mb;
  OptimizerOptions opt;
  opt.max_iterations = 2;
  const auto r = minimize(mb, Geometry::cartesian({-0.7, 1.3}), opt);
  EXPECT_FALSE(r.converged);
  EXPECT_EQ(r.iterations, 2);
}

TEST(Minimize, MolecularGeometryRelaxesToSpringRestLengths) {
  Springs model{{Spring{0, 1, 0.5, 1.8}, Spring{0, 2, 0.5, 1.8}, Spring{1, 2, 0.1, 2.9}}};
  Geometry g{{8, 1, 1}, Eigen::VectorXd(9)};
  g.coords << 0.0, 0.0, 0.0, 1.6, 0.9, 0.1, -1.3, 1.2, -0.2;
  OptimizerOptions opt;
  const auto r = minimize(model, g, opt);
  ASSERT_TRUE(r.converged);
  auto d = [&](int i, int j) {
    return (r.geometry.coords.segment<3>(3 * i) - r.geometry.coords.segment<3>(3 * j)).norm();
  };
  EXPECT_NEAR(d(0, 1), 1.8, 1e-4);
  EXPECT_NEAR(d(0, 2), 1.8, 1e-4);
  EXPECT_NEAR(d(1, 2), 2.9, 1e-4);
}

TEST(Minimize, StepsNeverExceedTrustRadius) {
  MullerBrown mb;
  OptimizerOptions opt;
  opt.trust_radius = 0.05;
  opt.max_trust_radius = 0.05;
  const auto r = minimize(mb, Geometry::cartesian({-0.7, 1.3}), opt);
  for (double s : r.step_norms) EXPECT_LE(s, 0.05 + 1e-12);
}

TEST(Bfgs, SatisfiesSecantCondition) {
  Eigen::MatrixXd h = Eigen::MatrixXd::Identity(3, 3);
  const Eigen::Vector3d s(0.1, -0.2, 0.05);
  const Eigen::Vector3d y(0.3, -0.1, 0.2);
  const Eigen::MatrixXd h1 = bfgs_update(h, s, y);
  EXPECT_TRUE((h1 * s).isApprox(y, 1e-12));
  EXPECT_TRUE(h1.isApprox(h1.transpose(), 1e-14));
}
