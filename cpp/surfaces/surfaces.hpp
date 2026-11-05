#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <vector>

#include "core/model.hpp"
#include "linalg/view.hpp"

namespace saddle::surfaces {

// Müller-Brown two-dimensional potential (standard parameters).
class MullerBrown {
 public:
  [[nodiscard]] Evaluation evaluate(const Geometry& geometry, const WarmStart& guess) const;
};

// Morse potential between atoms 0 and 1 of a two-atom Cartesian geometry:
// E = D * (1 - exp(-a (r - re)))^2.
class Morse {
 public:
  Morse(double depth, double alpha, double r_eq);
  [[nodiscard]] Evaluation evaluate(const Geometry& geometry, const WarmStart& guess) const;

 private:
  double depth_;
  double alpha_;
  double r_eq_;
};

// Quadratic surface E = 1/2 (x - x0)^T K (x - x0). K may be indefinite, which
// makes it a saddle of any order.
class Harmonic {
 public:
  Harmonic(Eigen::VectorXd center, Eigen::MatrixXd force_constants);
  [[nodiscard]] Evaluation evaluate(const Geometry& geometry, const WarmStart& guess) const;
  [[nodiscard]] const Eigen::MatrixXd& force_constants() const noexcept { return k_; }

 private:
  Eigen::VectorXd x0_;
  Eigen::MatrixXd k_;
};

// Pairwise harmonic springs E = sum 1/2 k (r_ij - r0)^2 on a Cartesian
// geometry. Translation and rotation invariant, so it exercises projection.
// A negative k makes that stretch a saddle direction.
struct Spring {
  int i;
  int j;
  double k;
  double r0;
};

class Springs {
 public:
  explicit Springs(std::vector<Spring> springs);
  [[nodiscard]] Evaluation evaluate(const Geometry& geometry, const WarmStart& guess) const;

 private:
  std::vector<Spring> springs_;
  int n_atoms_ = 0;
};

// Morse bond between atoms i and j: D (1 - exp(-a (r - re)))^2 - D.
struct MorseBond {
  int i;
  int j;
  double depth;
  double alpha;
  double r_eq;
};

// Sum of harmonic springs and Morse bonds on a Cartesian geometry. With two
// Morse bonds to a transferring atom and a spring holding the donor and
// acceptor apart, it is a bounded molecular double well with a real saddle.
class PairModel {
 public:
  PairModel(std::vector<Spring> springs, std::vector<MorseBond> morse);
  [[nodiscard]] Evaluation evaluate(const Geometry& geometry, const WarmStart& guess) const;

 private:
  Springs springs_;
  std::vector<MorseBond> morse_;
  int n_atoms_ = 0;
};

namespace detail {
// Deterministic Gaussian noise keyed on geometry bits, seed and component.
[[nodiscard]] Eigen::VectorXd geometry_noise(const Eigen::VectorXd& coords, double amplitude,
                                             std::uint64_t seed);
}  // namespace detail

// Adds reproducible Gaussian gradient noise with the given RMS amplitude.
// The same geometry always gets the same noise, like a converged but
// imperfect electronic-structure code.
template <ElectronicModel Inner>
class Noisy {
 public:
  Noisy(Inner inner, double amplitude, std::uint64_t seed)
      : inner_(std::move(inner)), amplitude_(amplitude), seed_(seed) {}

  [[nodiscard]] Evaluation evaluate(const Geometry& geometry, const WarmStart& guess) {
    Evaluation e = inner_.evaluate(geometry, guess);
    if (amplitude_ != 0.0) e.gradient += detail::geometry_noise(geometry.coords, amplitude_, seed_);
    return e;
  }

 private:
  Inner inner_;
  double amplitude_;
  std::uint64_t seed_;
};

// Adds an energy step when one coordinate crosses a threshold, mimicking a
// basis-rank or screening discontinuity.
template <ElectronicModel Inner>
class Discontinuous {
 public:
  Discontinuous(Inner inner, Eigen::Index coordinate, double threshold, double jump)
      : inner_(std::move(inner)), coordinate_(coordinate), threshold_(threshold), jump_(jump) {}

  [[nodiscard]] Evaluation evaluate(const Geometry& geometry, const WarmStart& guess) {
    Evaluation e = inner_.evaluate(geometry, guess);
    if (geometry.coords[coordinate_] > threshold_) e.energy += Hartree{jump_};
    return e;
  }

 private:
  Inner inner_;
  Eigen::Index coordinate_;
  double threshold_;
  double jump_;
};

}  // namespace saddle::surfaces
