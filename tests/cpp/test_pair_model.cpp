#include <gtest/gtest.h>

#include <Eigen/Dense>

#include "hess/fd_hessian.hpp"
#include "hess/vibrations.hpp"
#include "search/irc.hpp"
#include "search/optimizer.hpp"
#include "surfaces/surfaces.hpp"

using namespace saddle;
using namespace saddle::surfaces;

namespace {

constexpr double kAC = 6.5;

PairModel h_transfer() {
  return PairModel{{Spring{0, 2, 1.0, kAC}},
                   {MorseBond{0, 1, 0.15, 1.0, 1.8}, MorseBond{1, 2, 0.15, 1.0, 1.8}}};
}

Geometry at(double h_x, double h_y = 0.0) {
  Geometry g{{6, 1, 6}, Eigen::VectorXd::Zero(9)};
  g.coords[3] = h_x;
  g.coords[4] = h_y;
  g.coords[6] = kAC;
  return g;
}

const Eigen::Vector3d kMasses(12.0, 1.00782503223, 12.0);

}

static_assert(ElectronicModel<PairModel>);

TEST(PairModel, GradientMatchesFiniteDifference) {
  auto model = h_transfer();
  Geometry g = at(2.2, 0.3);
  g.coords[8] = 0.2;
  const Eigen::VectorXd analytic = model.evaluate(g, {}).gradient;
  for (Eigen::Index i = 0; i < 9; ++i) {
    Geometry p = g, m = g;
    p.coords[i] += 1e-6;
    m.coords[i] -= 1e-6;
    EXPECT_NEAR(analytic[i], (model.evaluate(p, {}).energy.value() - model.evaluate(m, {}).energy.value()) / 2e-6, 1e-8);
  }
}

TEST(PairModel, SymmetricDoubleWellHasOneImaginaryModeAtTheMiddle) {
  auto model = h_transfer();
  search::SaddleOptions opt;
  const auto ts = search::saddle_search(model, at(3.0, 0.05), opt);
  ASSERT_TRUE(ts.converged);
  EXPECT_NEAR(ts.geometry.coords[3] - ts.geometry.coords[0], 0.5 * (ts.geometry.coords[6] - ts.geometry.coords[0]), 1e-4);
  const auto vib = hess::analyze_vibrations(hess::estimate_hessian(model, ts.geometry, 1e-4).hessian,
                                            ts.geometry.coords, kMasses);
  EXPECT_EQ(vib.n_imaginary(1.0), 1);
}

TEST(PairModel, IrcConnectsTheTwoWells) {
  auto model = h_transfer();
  const auto ts = search::saddle_search(model, at(3.1, 0.02), search::SaddleOptions{});
  ASSERT_TRUE(ts.converged);
  const auto h = hess::estimate_hessian(model, ts.geometry, 1e-4).hessian;
  const auto r = search::irc(model, ts.geometry, h, kMasses, search::IrcOptions{});
  ASSERT_TRUE(r.forward.endpoint.converged);
  ASSERT_TRUE(r.reverse.endpoint.converged);
  auto ah = [](const Geometry& g) { return (g.coords.segment<3>(3) - g.coords.segment<3>(0)).norm(); };
  const double a = ah(r.forward.endpoint.geometry), b = ah(r.reverse.endpoint.geometry);
  EXPECT_NEAR(std::min(a, b), 1.8, 0.1);
  EXPECT_NEAR(std::max(a, b), kAC - 1.8, 0.1);
}
