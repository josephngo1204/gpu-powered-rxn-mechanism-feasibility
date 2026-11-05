#include <gtest/gtest.h>
#include <rapidcheck/gtest.h>

#include <cmath>
#include <functional>
#include <type_traits>
#include <vector>

#include "linalg/quantity.hpp"
#include "linalg/spin.hpp"
#include "linalg/view.hpp"

using namespace saddle;

TEST(Quantity, SameTagArithmetic) {
  const Hartree a{1.5};
  const Hartree b{0.5};
  EXPECT_DOUBLE_EQ((a + b).value(), 2.0);
  EXPECT_DOUBLE_EQ((a - b).value(), 1.0);
  EXPECT_DOUBLE_EQ((a * 2.0).value(), 3.0);
  EXPECT_LT(b, a);
}

TEST(Quantity, DifferentTagsDoNotMix) {
  static_assert(!std::is_convertible_v<Hartree, Bohr>);
  static_assert(!std::is_convertible_v<double, Hartree>);
  static_assert(!std::is_invocable_v<std::plus<>, Hartree, Bohr>);
}

TEST(Quantity, AngstromBoundaryConversion) {
  EXPECT_NEAR(to_angstrom(Bohr{1.0}), 0.529177210903, 1e-15);
  EXPECT_NEAR(bohr_from_angstrom(0.529177210903).value(), 1.0, 1e-15);
}

RC_GTEST_PROP(Quantity, AngstromRoundTrip, ()) {
  const double x = *rc::gen::inRange(-1'000'000'000, 1'000'000'000) * 1e-3;
  RC_ASSERT(std::abs(to_angstrom(bohr_from_angstrom(x)) - x) <= 1e-12 * (1.0 + std::abs(x)));
}

TEST(View, CarriesExtents) {
  std::vector<double> data(6, 0.0);
  MatrixView<double> m(data.data(), 2, 3);
  EXPECT_EQ(m.rows(), 2U);
  EXPECT_EQ(m.cols(), 3U);
  EXPECT_EQ(m.size(), 6U);
  m(1, 2) = 4.0;
  EXPECT_DOUBLE_EQ(data[5], 4.0);
}

TEST(View, CheckedAccessThrowsOutOfBounds) {
  std::vector<double> data(6, 0.0);
  MatrixView<double> m(data.data(), 2, 3);
  EXPECT_THROW((void)m.at(2, 0), ShapeError);
  EXPECT_THROW((void)m.at(0, 3), ShapeError);
}

TEST(View, RequireShapeRejectsMismatch) {
  std::vector<double> data(6, 0.0);
  MatrixView<const double> m(data.data(), 2, 3);
  EXPECT_NO_THROW(require_shape(m, 2, 3, "m"));
  EXPECT_THROW(require_shape(m, 3, 2, "m"), ShapeError);
}

TEST(View, BufferBackedViewRejectsUndersizedStorage) {
  std::vector<double> data(5, 0.0);
  EXPECT_THROW((void)make_matrix_view(data, 2, 3), ShapeError);
  std::vector<double> ok(6, 0.0);
  EXPECT_EQ(make_matrix_view(ok, 3, 2).rows(), 3U);
}

TEST(Spin, JKGradientScalesAreEncodedInTypes) {
  EXPECT_DOUBLE_EQ(SpinTraits<Restricted>::coulomb_gradient_scale, 2.0);
  EXPECT_DOUBLE_EQ(SpinTraits<Restricted>::exchange_gradient_scale, -1.0);
  EXPECT_DOUBLE_EQ(SpinTraits<Unrestricted>::coulomb_gradient_scale, 0.5);
  EXPECT_DOUBLE_EQ(SpinTraits<Unrestricted>::exchange_gradient_scale, -0.5);
  EXPECT_EQ(SpinTraits<Restricted>::n_density_matrices, 1);
  EXPECT_EQ(SpinTraits<Unrestricted>::n_density_matrices, 2);
}
