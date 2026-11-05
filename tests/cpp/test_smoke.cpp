#include <gtest/gtest.h>
#include <rapidcheck/gtest.h>

#include <Eigen/Dense>

#include "linalg/version.hpp"

TEST(Smoke, VersionIsZeroZeroOne) {
  EXPECT_EQ(saddle::kVersionMajor, 0);
  EXPECT_EQ(saddle::kVersionPatch, 1);
}

TEST(Smoke, EigenSolvesIdentity) {
  const Eigen::Matrix3d identity = Eigen::Matrix3d::Identity();
  const Eigen::Vector3d b(1.0, 2.0, 3.0);
  EXPECT_TRUE(identity.ldlt().solve(b).isApprox(b));
}

RC_GTEST_PROP(Smoke, AdditionCommutes, (int a, int b)) {
  RC_ASSERT(static_cast<long>(a) + b == static_cast<long>(b) + a);
}
