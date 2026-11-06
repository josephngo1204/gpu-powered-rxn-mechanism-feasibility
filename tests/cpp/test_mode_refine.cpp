#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>

#include "hess/mode_refine.hpp"
#include "surfaces/surfaces.hpp"

using namespace saddle;
using namespace saddle::hess;
using saddle::surfaces::Harmonic;
using saddle::surfaces::Morse;

namespace {

constexpr double kD = 0.15, kA = 1.0, kRe = 1.4, kR = 1.9;

double exact_curvature() {
  const double e = std::exp(-kA * (kR - kRe));
  return 2.0 * kD * kA * kA * e * (2.0 * e - 1.0);
}

Geometry stretched() {
  Geometry g{{1, 1}, Eigen::VectorXd::Zero(6)};
  g.coords[5] = kR;
  return g;
}

Eigen::VectorXd bond_direction() {
  Eigen::VectorXd v = Eigen::VectorXd::Zero(6);
  v[5] = 1.0;
  return v;
}

double error(Stencil s, double h) {
  Morse morse{kD, kA, kRe};
  return std::abs(directional_curvature(morse, stretched(), bond_direction(), h, s).curvature -
                  exact_curvature());
}

}

TEST(DirectionalCurvature, CentralIsSecondOrder) {
  const double ratio = error(Stencil::Central, 0.02) / error(Stencil::Central, 0.01);
  EXPECT_NEAR(ratio, 4.0, 0.2);
}

TEST(DirectionalCurvature, FivePointIsFourthOrder) {
  const double ratio = error(Stencil::FivePoint, 0.08) / error(Stencil::FivePoint, 0.04);
  EXPECT_NEAR(ratio, 16.0, 1.0);
}

TEST(DirectionalCurvature, FivePointBeatsCentralAtSameStep) {
  EXPECT_LT(error(Stencil::FivePoint, 0.01), 0.01 * error(Stencil::Central, 0.01));
}

TEST(DirectionalCurvature, GradientCounts) {
  Morse morse{kD, kA, kRe};
  EXPECT_EQ(directional_curvature(morse, stretched(), bond_direction(), 0.01, Stencil::Central).n_gradients, 2);
  EXPECT_EQ(directional_curvature(morse, stretched(), bond_direction(), 0.01, Stencil::FivePoint).n_gradients, 4);
}

TEST(DirectionalCurvature, NormalizesDirection) {
  Eigen::MatrixXd k(2, 2);
  k << 2.0, 0.0, 0.0, 5.0;
  Harmonic h{Eigen::Vector2d::Zero(), k};
  const auto r = directional_curvature(h, Geometry::cartesian({0.0, 0.0}), Eigen::Vector2d(0.0, 3.0),
                                       1e-3, Stencil::Central);
  EXPECT_NEAR(r.curvature, 5.0, 1e-9);
  EXPECT_TRUE(r.hessian_vector.isApprox(Eigen::Vector2d(0.0, 5.0), 1e-9));
}

TEST(RefineModes, ReportsCentralFivePointDifferencePerMode) {
  Morse morse{kD, kA, kRe};
  Eigen::MatrixXd modes(6, 1);
  modes.col(0) = bond_direction();
  const auto refined = refine_modes(morse, stretched(), modes, 0.02);
  ASSERT_EQ(refined.size(), 1U);
  EXPECT_NEAR(refined[0].refined, exact_curvature(), 1e-6);
  EXPECT_NEAR(refined[0].richardson_correction, refined[0].refined - refined[0].central, 1e-15);
  EXPECT_GT(std::abs(refined[0].richardson_correction), 1e-6);
  EXPECT_EQ(refined[0].n_gradients, 4);
}

TEST(DirectionalCurvature, RejectsBadInput) {
  Morse morse{kD, kA, kRe};
  EXPECT_THROW((void)directional_curvature(morse, stretched(), Eigen::VectorXd::Zero(6), 0.01,
                                           Stencil::Central),
               std::invalid_argument);
  EXPECT_THROW((void)directional_curvature(morse, stretched(), Eigen::VectorXd::Ones(3), 0.01,
                                           Stencil::Central),
               ShapeError);
}
