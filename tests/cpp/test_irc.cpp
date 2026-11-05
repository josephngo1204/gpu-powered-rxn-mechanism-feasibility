#include <gtest/gtest.h>

#include <Eigen/Dense>

#include "hess/fd_hessian.hpp"
#include "search/irc.hpp"
#include "surfaces/surfaces.hpp"

using namespace saddle;
using namespace saddle::search;
using saddle::surfaces::MullerBrown;

namespace {

IrcOptions mb_options() {
  IrcOptions o;
  o.step = 0.02;
  o.max_steps = 400;
  o.switch_gradient = 1.0;
  return o;
}

bool near(const Geometry& g, double x, double y) {
  return std::abs(g.coords[0] - x) < 2e-3 && std::abs(g.coords[1] - y) < 2e-3;
}

void expect_monotone_descent(const IrcBranch& b) {
  for (std::size_t i = 1; i < b.energies.size(); ++i) EXPECT_LT(b.energies[i], b.energies[i - 1]);
}

struct IrcCase {
  double ts_x, ts_y, a_x, a_y, b_x, b_y;
};

}  // namespace

class MullerBrownIrc : public ::testing::TestWithParam<IrcCase> {};

TEST_P(MullerBrownIrc, ConnectsTheAdjacentMinima) {
  const auto c = GetParam();
  MullerBrown mb;
  const auto ts = Geometry::cartesian({c.ts_x, c.ts_y});
  const auto h = hess::estimate_hessian(mb, ts, 1e-5).hessian;
  const auto r = irc(mb, ts, h, Eigen::VectorXd(), mb_options());
  ASSERT_TRUE(r.forward.endpoint.converged);
  ASSERT_TRUE(r.reverse.endpoint.converged);
  const auto& f = r.forward.endpoint.geometry;
  const auto& b = r.reverse.endpoint.geometry;
  const bool ab = near(f, c.a_x, c.a_y) && near(b, c.b_x, c.b_y);
  const bool ba = near(f, c.b_x, c.b_y) && near(b, c.a_x, c.a_y);
  EXPECT_TRUE(ab || ba) << "forward (" << f.coords.transpose() << ") reverse (" << b.coords.transpose() << ")";
  expect_monotone_descent(r.forward);
  expect_monotone_descent(r.reverse);
  EXPECT_GT(r.forward.path.size(), 3U);
}

INSTANTIATE_TEST_SUITE_P(Known, MullerBrownIrc,
                         ::testing::Values(IrcCase{0.212487, 0.292988, 0.623, 0.028, -0.050, 0.467},
                                           IrcCase{-0.822001, 0.624313, -0.558, 1.442, -0.050, 0.467}));

TEST(Irc, ForwardFollowsPositiveModeSign) {
  MullerBrown mb;
  const auto ts = Geometry::cartesian({0.212487, 0.292988});
  const auto h = hess::estimate_hessian(mb, ts, 1e-5).hessian;
  const auto r = irc(mb, ts, h, Eigen::VectorXd(), mb_options());
  const Eigen::VectorXd d = r.forward.path[1].coords - ts.coords;
  EXPECT_GT(d.dot(r.transition_mode), 0.0);
}

TEST(Irc, RejectsMinimumAsStart) {
  MullerBrown mb;
  const auto m = Geometry::cartesian({0.623499, 0.028038});
  const auto h = hess::estimate_hessian(mb, m, 1e-5).hessian;
  EXPECT_THROW((void)irc(mb, m, h, Eigen::VectorXd(), mb_options()), std::invalid_argument);
}

TEST(Irc, RejectsShapeMismatch) {
  MullerBrown mb;
  EXPECT_THROW((void)irc(mb, Geometry::cartesian({0.2, 0.3}), Eigen::MatrixXd::Identity(3, 3),
                         Eigen::VectorXd(), mb_options()),
               ShapeError);
}
