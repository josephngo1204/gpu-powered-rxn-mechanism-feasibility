#include <gtest/gtest.h>

#include <Eigen/Dense>

#include "hess/noise.hpp"
#include "surfaces/surfaces.hpp"

using namespace saddle;
using namespace saddle::hess;
using saddle::surfaces::Morse;
using saddle::surfaces::MullerBrown;
using saddle::surfaces::Noisy;

namespace {
const Eigen::Vector2d kDirection = Eigen::Vector2d(1.0, 1.0).normalized();
}

class NoiseRecovery : public ::testing::TestWithParam<double> {};

TEST_P(NoiseRecovery, RecoversInjectedNoiseWithinFactorTwo) {
  const double eps = GetParam();
  Noisy<MullerBrown> model{MullerBrown{}, eps, 11};
  const auto est = probe_noise(model, Geometry::cartesian({0.1, 0.4}), kDirection, 1e-3);
  EXPECT_GT(est.epsilon, eps / 2.0);
  EXPECT_LT(est.epsilon, eps * 2.0);
  EXPECT_EQ(est.n_gradients, 16);
}

INSTANTIATE_TEST_SUITE_P(Levels, NoiseRecovery, ::testing::Values(1e-5, 1e-6, 1e-7, 1e-8));

TEST(NoiseProbe, CleanSurfaceReportsTinyNoise) {
  MullerBrown clean;
  const auto est = probe_noise(clean, Geometry::cartesian({0.1, 0.4}), kDirection, 1e-3);
  EXPECT_LT(est.epsilon, 1e-9);
}

TEST(NoiseProbe, RejectsBadArguments) {
  MullerBrown mb;
  const auto g = Geometry::cartesian({0.1, 0.4});
  EXPECT_THROW((void)probe_noise(mb, g, Eigen::Vector3d::UnitX(), 1e-3), ShapeError);
  EXPECT_THROW((void)probe_noise(mb, g, Eigen::Vector2d::Zero(), 1e-3), std::invalid_argument);
  EXPECT_THROW((void)probe_noise(mb, g, kDirection, -1.0), std::invalid_argument);
  EXPECT_THROW((void)probe_noise(mb, g, kDirection, 1e-3, 6, 6), std::invalid_argument);
}

struct TableRow {
  double eps, h_opt, dk, freq;
};

class SpecTable : public ::testing::TestWithParam<TableRow> {};

TEST_P(SpecTable, ReproducesDesignEstimate) {
  const auto row = GetParam();
  const double h = optimal_step(row.eps, 2.0);
  EXPECT_NEAR(h, row.h_opt, 0.05 * row.h_opt);
  const double dk = fd_error(row.eps, h, 2.0);
  EXPECT_NEAR(dk, row.dk, 0.06 * row.dk);
  EXPECT_NEAR(noise_frequency_cm(dk, 12.0), row.freq, 0.6);
}

INSTANTIATE_TEST_SUITE_P(Rows, SpecTable,
                         ::testing::Values(TableRow{1e-6, 0.011, 1.3e-4, 17.0},
                                           TableRow{1e-7, 0.0053, 2.9e-5, 8.0},
                                           TableRow{1e-8, 0.0025, 6e-6, 4.0}));

TEST(OptimalStep, MinimizesTotalError) {
  const double eps = 1e-7;
  const double h = optimal_step(eps, 2.0);
  EXPECT_LT(fd_error(eps, h, 2.0), fd_error(eps, 0.8 * h, 2.0));
  EXPECT_LT(fd_error(eps, h, 2.0), fd_error(eps, 1.25 * h, 2.0));
}

TEST(NoiseProbe, MorseLineScanWorksOnCartesianModel) {
  Noisy<Morse> model{Morse{0.15, 1.0, 1.4}, 1e-7, 5};
  Geometry g{{1, 1}, Eigen::VectorXd::Zero(6)};
  g.coords[5] = 1.5;
  Eigen::VectorXd u = Eigen::VectorXd::Zero(6);
  u[5] = 1.0;
  const auto est = probe_noise(model, g, u, 1e-3);
  EXPECT_GT(est.epsilon, 0.5e-7);
  EXPECT_LT(est.epsilon, 2e-7);
}
