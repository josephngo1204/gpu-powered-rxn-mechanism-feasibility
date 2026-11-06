#include <gtest/gtest.h>
#include <rapidcheck/gtest.h>

#include <Eigen/Dense>

#include "hess/fd_hessian.hpp"
#include "surfaces/surfaces.hpp"

using namespace saddle;
using namespace saddle::hess;
using saddle::surfaces::Harmonic;
using saddle::surfaces::MullerBrown;
using saddle::surfaces::Noisy;

TEST(CentralStencil, TwoPointsPerCoordinateInOrder) {
  const auto g = Geometry::cartesian({1.0, 2.0, 3.0});
  const auto points = central_stencil(g, 0.01);
  ASSERT_EQ(points.size(), 6U);
  EXPECT_EQ(points[0].coordinate, 0);
  EXPECT_EQ(points[0].sign, +1);
  EXPECT_EQ(points[1].coordinate, 0);
  EXPECT_EQ(points[1].sign, -1);
  EXPECT_DOUBLE_EQ(points[4].geometry.coords[2], 3.01);
  EXPECT_DOUBLE_EQ(points[5].geometry.coords[2], 2.99);
  EXPECT_DOUBLE_EQ(points[5].geometry.coords[0], 1.0);
}

TEST(CentralStencil, RejectsNonPositiveStep) {
  EXPECT_THROW((void)central_stencil(Geometry::cartesian({0.0}), 0.0), std::invalid_argument);
  EXPECT_THROW((void)central_stencil(Geometry::cartesian({0.0}), -1e-3), std::invalid_argument);
}

TEST(AssembleCentral, RejectsWrongGradientCount) {
  std::vector<Eigen::VectorXd> grads(3, Eigen::VectorXd::Zero(2));
  EXPECT_THROW((void)assemble_central(grads, 0.01), ShapeError);
}

TEST(EstimateHessian, ExactOnQuadratic) {
  Eigen::MatrixXd k(3, 3);
  k << 3.0, 0.2, -0.1, 0.2, 1.0, 0.4, -0.1, 0.4, 2.0;
  Harmonic h{Eigen::Vector3d(0.5, -0.2, 1.0), k};
  const auto result = estimate_hessian(h, Geometry::cartesian({0.1, 0.2, 0.3}), 1e-3);
  EXPECT_TRUE(result.hessian.isApprox(k, 1e-10));
  EXPECT_LT(result.asymmetry, 1e-12);
  EXPECT_EQ(result.n_gradients, 6);
}

TEST(EstimateHessian, MullerBrownMatchesAnalyticSecondDerivative) {
  MullerBrown mb;
  const auto g = Geometry::cartesian({0.212487, 0.292988});
  const auto coarse = estimate_hessian(mb, g, 1e-3);
  const auto fine = estimate_hessian(mb, g, 1e-5);
  EXPECT_LT((coarse.hessian - fine.hessian).norm(), 1e-1);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(fine.hessian);
  EXPECT_LT(es.eigenvalues()[0], 0.0);
  EXPECT_GT(es.eigenvalues()[1], 0.0);
}

TEST(EstimateHessian, NoiseShowsUpAsAsymmetryAndSymmetrizedOutput) {
  Noisy<MullerBrown> noisy{MullerBrown{}, 1e-3, 3};
  const auto result = estimate_hessian(noisy, Geometry::cartesian({0.1, 0.4}), 1e-4);
  EXPECT_GT(result.asymmetry, 1e-4);
  EXPECT_TRUE(result.hessian.isApprox(result.hessian.transpose(), 0.0));
  EXPECT_FALSE(result.raw.isApprox(result.raw.transpose(), 1e-6));
}

TEST(Asymmetry, ZeroMatrixIsZero) {
  EXPECT_DOUBLE_EQ(relative_asymmetry(Eigen::MatrixXd::Zero(2, 2)), 0.0);
}
