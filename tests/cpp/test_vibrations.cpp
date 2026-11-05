#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>

#include "hess/fd_hessian.hpp"
#include "hess/vibrations.hpp"
#include "surfaces/surfaces.hpp"

using namespace saddle;
using namespace saddle::hess;
using saddle::surfaces::Morse;
using saddle::surfaces::Spring;
using saddle::surfaces::Springs;

namespace {

constexpr double kHartreeToWavenumber = 219474.6313632;
constexpr double kAmuToElectronMass = 1822.888486209;
constexpr double kHydrogenMass = 1.00782503223;

// Bent triatomic at a spring equilibrium (all springs at rest length).
Geometry triangle() {
  Geometry g{{8, 1, 1}, Eigen::VectorXd(9)};
  g.coords << 0.0, 0.0, 0.0, 1.43, 1.11, 0.0, -1.43, 1.11, 0.0;
  return g;
}

Springs triangle_springs(double k_hh) {
  const auto g = triangle();
  auto dist = [&](int i, int j) {
    return (g.coords.segment<3>(3 * i) - g.coords.segment<3>(3 * j)).norm();
  };
  return Springs{{Spring{0, 1, 0.5, dist(0, 1)}, Spring{0, 2, 0.5, dist(0, 2)},
                  Spring{1, 2, k_hh, dist(1, 2)}}};
}

const Eigen::Vector3d kWaterMasses(15.99491461957, kHydrogenMass, kHydrogenMass);

}  // namespace

TEST(Springs, GradientMatchesFiniteDifference) {
  auto model = triangle_springs(0.1);
  Geometry g = triangle();
  g.coords[3] += 0.1;
  g.coords[7] -= 0.05;
  const Eigen::VectorXd analytic = model.evaluate(g, {}).gradient;
  for (Eigen::Index i = 0; i < 9; ++i) {
    Geometry p = g, m = g;
    p.coords[i] += 1e-6;
    m.coords[i] -= 1e-6;
    const double fd =
        (model.evaluate(p, {}).energy.value() - model.evaluate(m, {}).energy.value()) / 2e-6;
    EXPECT_NEAR(analytic[i], fd, 1e-8);
  }
}

TEST(Vibrations, DiatomicMorseMatchesAnalyticFrequency) {
  Morse morse{0.15, 1.0, 1.4};
  Geometry g{{1, 1}, Eigen::VectorXd::Zero(6)};
  g.coords[5] = 1.4;
  const auto hess = estimate_hessian(morse, g, 1e-4);
  const auto vib = analyze_vibrations(hess.hessian, g.coords, Eigen::Vector2d(kHydrogenMass, kHydrogenMass));
  ASSERT_TRUE(vib.linear);
  ASSERT_EQ(vib.n_external, 5);
  ASSERT_EQ(vib.frequencies_cm.size(), 1);
  const double mu = 0.5 * kHydrogenMass * kAmuToElectronMass;
  const double expected = 1.0 * std::sqrt(2.0 * 0.15 / mu) * kHartreeToWavenumber;
  EXPECT_NEAR(vib.frequencies_cm[0], expected, 0.5);
  EXPECT_LT(vib.max_external_residual_cm, 5.0);
}

TEST(Vibrations, BentTriatomicHasThreeModesAndCleanProjection) {
  auto model = triangle_springs(0.1);
  const auto g = triangle();
  const auto hess = estimate_hessian(model, g, 1e-3);
  const auto vib = analyze_vibrations(hess.hessian, g.coords, kWaterMasses);
  EXPECT_FALSE(vib.linear);
  EXPECT_EQ(vib.n_external, 6);
  ASSERT_EQ(vib.frequencies_cm.size(), 3);
  for (Eigen::Index i = 0; i < 3; ++i) EXPECT_GT(vib.frequencies_cm[i], 100.0);
  EXPECT_LT(vib.max_external_residual_cm, 5.0);
  EXPECT_LT(vib.translation_sum_rule, 1e-8);
  // Frequencies are sorted ascending.
  EXPECT_LE(vib.frequencies_cm[0], vib.frequencies_cm[1]);
  EXPECT_LE(vib.frequencies_cm[1], vib.frequencies_cm[2]);
}

TEST(Vibrations, ModesAreOrthonormalInMassWeightedSpace) {
  auto model = triangle_springs(0.1);
  const auto g = triangle();
  const auto vib = analyze_vibrations(estimate_hessian(model, g, 1e-3).hessian, g.coords, kWaterMasses);
  const Eigen::MatrixXd overlap = vib.modes_mass_weighted.transpose() * vib.modes_mass_weighted;
  EXPECT_TRUE(overlap.isApprox(Eigen::MatrixXd::Identity(3, 3), 1e-10));
}

TEST(Vibrations, NegativeSpringGivesOneImaginaryMode) {
  auto model = triangle_springs(-0.05);
  const auto g = triangle();
  const auto vib = analyze_vibrations(estimate_hessian(model, g, 1e-3).hessian, g.coords, kWaterMasses);
  ASSERT_EQ(vib.frequencies_cm.size(), 3);
  EXPECT_LT(vib.frequencies_cm[0], -100.0);
  EXPECT_GT(vib.frequencies_cm[1], 0.0);
  EXPECT_EQ(vib.n_imaginary(0.0), 1);
}

TEST(Vibrations, UnprojectedTranslationalErrorIsReported) {
  auto model = triangle_springs(0.1);
  const auto g = triangle();
  Eigen::MatrixXd h = estimate_hessian(model, g, 1e-3).hessian;
  h += 1e-3 * Eigen::MatrixXd::Identity(9, 9);  // breaks translation invariance
  const auto vib = analyze_vibrations(h, g.coords, kWaterMasses);
  EXPECT_GT(vib.max_external_residual_cm, 5.0);
  EXPECT_GT(vib.translation_sum_rule, 1e-4);
}

TEST(Vibrations, RejectsShapeMismatch) {
  EXPECT_THROW((void)analyze_vibrations(Eigen::MatrixXd::Zero(6, 6), Eigen::VectorXd::Zero(6),
                                        Eigen::Vector3d::Ones()),
               ShapeError);
  EXPECT_THROW((void)analyze_vibrations(Eigen::MatrixXd::Zero(5, 6), Eigen::VectorXd::Zero(6),
                                        Eigen::Vector2d::Ones()),
               ShapeError);
  EXPECT_THROW((void)analyze_vibrations(Eigen::MatrixXd::Zero(6, 6), Eigen::VectorXd::Zero(6),
                                        Eigen::Vector2d(1.0, -1.0)),
               std::invalid_argument);
}
