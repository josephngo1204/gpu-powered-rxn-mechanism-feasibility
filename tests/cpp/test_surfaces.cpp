#include <gtest/gtest.h>
#include <rapidcheck/gtest.h>

#include <Eigen/Dense>
#include <cmath>

#include "core/model.hpp"
#include "surfaces/surfaces.hpp"

using namespace saddle;
using namespace saddle::surfaces;

static_assert(ElectronicModel<MullerBrown>);
static_assert(ElectronicModel<Morse>);
static_assert(ElectronicModel<Harmonic>);
static_assert(ElectronicModel<Noisy<MullerBrown>>);
static_assert(ElectronicModel<Discontinuous<Harmonic>>);

namespace {

template <ElectronicModel Model>
Eigen::VectorXd fd_gradient(Model& model, const Geometry& g, double h) {
  Eigen::VectorXd out(g.coords.size());
  for (Eigen::Index i = 0; i < g.coords.size(); ++i) {
    Geometry plus = g;
    Geometry minus = g;
    plus.coords[i] += h;
    minus.coords[i] -= h;
    out[i] = (model.evaluate(plus, {}).energy.value() - model.evaluate(minus, {}).energy.value()) /
             (2.0 * h);
  }
  return out;
}

double coord(double lo, double hi) {
  return lo + (hi - lo) * (*rc::gen::inRange(0, 1'000'000)) * 1e-6;
}

}

TEST(MullerBrown, KnownMinimumEnergy) {
  MullerBrown mb;
  const auto e = mb.evaluate(Geometry::cartesian({-0.558224, 1.441726}), {});
  EXPECT_NEAR(e.energy.value(), -146.69951720, 1e-5);
  EXPECT_LT(e.gradient.norm(), 1e-2);
  EXPECT_TRUE(e.diagnostics.converged);
}

TEST(MullerBrown, KnownSaddleIsStationary) {
  MullerBrown mb;
  const auto e = mb.evaluate(Geometry::cartesian({0.212487, 0.292988}), {});
  EXPECT_NEAR(e.energy.value(), -72.24894, 1e-4);
  EXPECT_LT(e.gradient.norm(), 1e-3);
}

RC_GTEST_PROP(MullerBrown, AnalyticGradientMatchesFiniteDifference, ()) {
  MullerBrown mb;
  const auto g = Geometry::cartesian({coord(-1.5, 1.2), coord(-0.3, 2.0)});
  const Eigen::VectorXd analytic = mb.evaluate(g, {}).gradient;
  const Eigen::VectorXd numeric = fd_gradient(mb, g, 1e-5);
  RC_ASSERT((analytic - numeric).norm() <= 1e-4 * (1.0 + analytic.norm()));
}

TEST(Morse, MinimumAtEquilibriumDistance) {
  Morse morse{0.15, 1.0, 1.4};
  Geometry g{{1, 1}, Eigen::VectorXd::Zero(6)};
  g.coords[5] = 1.4;
  const auto e = morse.evaluate(g, {});
  EXPECT_NEAR(e.energy.value(), 0.0, 1e-14);
  EXPECT_LT(e.gradient.norm(), 1e-12);
}

TEST(Morse, DissociationLimit) {
  Morse morse{0.15, 1.0, 1.4};
  Geometry g{{1, 1}, Eigen::VectorXd::Zero(6)};
  g.coords[5] = 60.0;
  EXPECT_NEAR(morse.evaluate(g, {}).energy.value(), 0.15, 1e-12);
}

RC_GTEST_PROP(Morse, AnalyticGradientMatchesFiniteDifference, ()) {
  Morse morse{0.15, 1.0, 1.4};
  Geometry g{{1, 1}, Eigen::VectorXd::Zero(6)};
  for (Eigen::Index i = 0; i < 6; ++i) g.coords[i] = coord(-1.5, 1.5);
  g.coords[5] += 2.0;
  const Eigen::VectorXd analytic = morse.evaluate(g, {}).gradient;
  const Eigen::VectorXd numeric = fd_gradient(morse, g, 1e-5);
  RC_ASSERT((analytic - numeric).norm() <= 1e-7);
}

RC_GTEST_PROP(Morse, GradientHasNoNetForce, ()) {
  Morse morse{0.15, 1.0, 1.4};
  Geometry g{{1, 1}, Eigen::VectorXd::Zero(6)};
  for (Eigen::Index i = 0; i < 6; ++i) g.coords[i] = coord(-2.0, 2.0);
  g.coords[5] += 3.0;
  const Eigen::VectorXd grad = morse.evaluate(g, {}).gradient;
  RC_ASSERT((grad.head<3>() + grad.tail<3>()).norm() <= 1e-12);
}

TEST(Harmonic, EnergyAndGradient) {
  Eigen::MatrixXd k(2, 2);
  k << 2.0, 0.5, 0.5, 1.0;
  Harmonic h{Eigen::Vector2d(1.0, -1.0), k};
  const auto e = h.evaluate(Geometry::cartesian({2.0, 0.0}), {});
  EXPECT_DOUBLE_EQ(e.energy.value(), 2.0);
  EXPECT_DOUBLE_EQ(e.gradient[0], 2.5);
  EXPECT_DOUBLE_EQ(e.gradient[1], 1.5);
}

TEST(Harmonic, RejectsAsymmetricForceConstants) {
  Eigen::MatrixXd k(2, 2);
  k << 1.0, 0.5, 0.0, 1.0;
  EXPECT_THROW((Harmonic{Eigen::Vector2d::Zero(), k}), std::invalid_argument);
}

TEST(Harmonic, RejectsWrongDimension) {
  Harmonic h{Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity()};
  EXPECT_THROW((void)h.evaluate(Geometry::cartesian({1.0, 2.0, 3.0}), {}), ShapeError);
}

TEST(Noisy, NoiseIsDeterministicPerGeometryWithRequestedAmplitude) {
  Noisy<MullerBrown> noisy{MullerBrown{}, 1e-6, 42};
  MullerBrown clean;
  const auto g = Geometry::cartesian({0.1, 0.4});
  const auto a = noisy.evaluate(g, {});
  const auto b = noisy.evaluate(g, {});
  EXPECT_EQ(a.gradient, b.gradient);

  double sum_sq = 0.0;
  int n = 0;
  for (int i = 0; i < 2000; ++i) {
    const auto gi = Geometry::cartesian({0.1 + 1e-3 * i, 0.4});
    const Eigen::VectorXd d = noisy.evaluate(gi, {}).gradient - clean.evaluate(gi, {}).gradient;
    sum_sq += d.squaredNorm();
    n += 2;
  }
  EXPECT_NEAR(std::sqrt(sum_sq / n), 1e-6, 1e-7);
}

TEST(Noisy, ZeroAmplitudeIsClean) {
  Noisy<MullerBrown> noisy{MullerBrown{}, 0.0, 1};
  MullerBrown clean;
  const auto g = Geometry::cartesian({0.3, 0.9});
  EXPECT_EQ(noisy.evaluate(g, {}).gradient, clean.evaluate(g, {}).gradient);
}

TEST(Discontinuous, EnergyJumpsAcrossThreshold) {
  Discontinuous<Harmonic> d{Harmonic{Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity()}, 0, 0.5,
                            1e-3};
  const double below = d.evaluate(Geometry::cartesian({0.5 - 1e-9, 0.0}), {}).energy.value();
  const double above = d.evaluate(Geometry::cartesian({0.5 + 1e-9, 0.0}), {}).energy.value();
  EXPECT_NEAR(above - below, 1e-3, 1e-8);
}
