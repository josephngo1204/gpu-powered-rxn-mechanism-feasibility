#include "surfaces/surfaces.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>

namespace saddle::surfaces {

namespace {

void require_dimension(const Geometry& g, Eigen::Index n, const char* surface) {
  if (g.dimension() != n) {
    throw ShapeError(std::string(surface) + ": expected " + std::to_string(n) +
                     " coordinates, got " + std::to_string(g.dimension()));
  }
}

constexpr std::array<double, 4> kA{-200.0, -100.0, -170.0, 15.0};
constexpr std::array<double, 4> ka{-1.0, -1.0, -6.5, 0.7};
constexpr std::array<double, 4> kb{0.0, 0.0, 11.0, 0.6};
constexpr std::array<double, 4> kc{-10.0, -10.0, -6.5, 0.7};
constexpr std::array<double, 4> kx0{1.0, 0.0, -0.5, -1.0};
constexpr std::array<double, 4> ky0{0.0, 0.5, 1.5, 1.0};

std::uint64_t splitmix64(std::uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

double to_unit_interval(std::uint64_t x) {
  // (0, 1], never zero so log() is finite.
  return (static_cast<double>(x >> 11) + 1.0) * 0x1.0p-53;
}

}  // namespace

Evaluation MullerBrown::evaluate(const Geometry& geometry, const WarmStart& /*guess*/) const {
  require_dimension(geometry, 2, "MullerBrown");
  const double x = geometry.coords[0];
  const double y = geometry.coords[1];
  double e = 0.0;
  double gx = 0.0;
  double gy = 0.0;
  for (std::size_t k = 0; k < 4; ++k) {
    const double dx = x - kx0[k];
    const double dy = y - ky0[k];
    const double term = kA[k] * std::exp(ka[k] * dx * dx + kb[k] * dx * dy + kc[k] * dy * dy);
    e += term;
    gx += term * (2.0 * ka[k] * dx + kb[k] * dy);
    gy += term * (kb[k] * dx + 2.0 * kc[k] * dy);
  }
  return Evaluation{Hartree{e}, Eigen::Vector2d(gx, gy), {}};
}

Morse::Morse(double depth, double alpha, double r_eq) : depth_(depth), alpha_(alpha), r_eq_(r_eq) {
  if (!(depth > 0.0 && alpha > 0.0 && r_eq > 0.0)) {
    throw std::invalid_argument("Morse parameters must be positive");
  }
}

Evaluation Morse::evaluate(const Geometry& geometry, const WarmStart& /*guess*/) const {
  require_dimension(geometry, 6, "Morse");
  const Eigen::Vector3d d = geometry.coords.segment<3>(3) - geometry.coords.segment<3>(0);
  const double r = d.norm();
  if (r == 0.0) throw std::domain_error("Morse: atoms coincide");
  const double ex = std::exp(-alpha_ * (r - r_eq_));
  const double energy = depth_ * (1.0 - ex) * (1.0 - ex);
  const double de_dr = 2.0 * depth_ * alpha_ * ex * (1.0 - ex);
  const Eigen::Vector3d g1 = de_dr * d / r;
  Eigen::VectorXd gradient(6);
  gradient << -g1, g1;
  return Evaluation{Hartree{energy}, gradient, {}};
}

Harmonic::Harmonic(Eigen::VectorXd center, Eigen::MatrixXd force_constants)
    : x0_(std::move(center)), k_(std::move(force_constants)) {
  if (k_.rows() != x0_.size() || k_.cols() != x0_.size()) {
    throw ShapeError("Harmonic: force-constant matrix does not match center dimension");
  }
  if (!k_.isApprox(k_.transpose(), 1e-12)) {
    throw std::invalid_argument("Harmonic: force-constant matrix must be symmetric");
  }
}

Evaluation Harmonic::evaluate(const Geometry& geometry, const WarmStart& /*guess*/) const {
  require_dimension(geometry, x0_.size(), "Harmonic");
  const Eigen::VectorXd dx = geometry.coords - x0_;
  const Eigen::VectorXd g = k_ * dx;
  return Evaluation{Hartree{0.5 * dx.dot(g)}, g, {}};
}

Springs::Springs(std::vector<Spring> springs) : springs_(std::move(springs)) {
  for (const auto& s : springs_) {
    if (s.i < 0 || s.j < 0 || s.i == s.j || !(s.r0 > 0.0)) {
      throw std::invalid_argument("Springs: invalid spring definition");
    }
    n_atoms_ = std::max({n_atoms_, s.i + 1, s.j + 1});
  }
}

Evaluation Springs::evaluate(const Geometry& geometry, const WarmStart& /*guess*/) const {
  if (geometry.dimension() < 3 * n_atoms_ || geometry.dimension() % 3 != 0) {
    throw ShapeError("Springs: geometry has too few Cartesian coordinates");
  }
  double energy = 0.0;
  Eigen::VectorXd gradient = Eigen::VectorXd::Zero(geometry.dimension());
  for (const auto& s : springs_) {
    const Eigen::Vector3d d = geometry.coords.segment<3>(3 * s.j) - geometry.coords.segment<3>(3 * s.i);
    const double r = d.norm();
    if (r == 0.0) throw std::domain_error("Springs: atoms coincide");
    const double stretch = r - s.r0;
    energy += 0.5 * s.k * stretch * stretch;
    const Eigen::Vector3d f = s.k * stretch * d / r;
    gradient.segment<3>(3 * s.j) += f;
    gradient.segment<3>(3 * s.i) -= f;
  }
  return Evaluation{Hartree{energy}, gradient, {}};
}

namespace detail {

Eigen::VectorXd geometry_noise(const Eigen::VectorXd& coords, double amplitude,
                               std::uint64_t seed) {
  std::uint64_t key = splitmix64(seed);
  for (Eigen::Index i = 0; i < coords.size(); ++i) {
    key = splitmix64(key ^ std::bit_cast<std::uint64_t>(coords[i]));
  }
  Eigen::VectorXd noise(coords.size());
  for (Eigen::Index i = 0; i < coords.size(); ++i) {
    const std::uint64_t a = splitmix64(key + 2 * static_cast<std::uint64_t>(i));
    const std::uint64_t b = splitmix64(key + 2 * static_cast<std::uint64_t>(i) + 1);
    const double u1 = to_unit_interval(a);
    const double u2 = to_unit_interval(b);
    noise[i] = amplitude * std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * std::numbers::pi * u2);
  }
  return noise;
}

}  // namespace detail

}  // namespace saddle::surfaces
