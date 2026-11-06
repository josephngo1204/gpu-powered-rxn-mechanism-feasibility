#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <vector>

#include "core/model.hpp"
#include "linalg/view.hpp"

namespace saddle::surfaces {

class MullerBrown {
 public:
  [[nodiscard]] Evaluation evaluate(const Geometry& geometry, const WarmStart& guess) const;
};

class Morse {
 public:
  Morse(double depth, double alpha, double r_eq);
  [[nodiscard]] Evaluation evaluate(const Geometry& geometry, const WarmStart& guess) const;

 private:
  double depth_;
  double alpha_;
  double r_eq_;
};

class Harmonic {
 public:
  Harmonic(Eigen::VectorXd center, Eigen::MatrixXd force_constants);
  [[nodiscard]] Evaluation evaluate(const Geometry& geometry, const WarmStart& guess) const;
  [[nodiscard]] const Eigen::MatrixXd& force_constants() const noexcept { return k_; }

 private:
  Eigen::VectorXd x0_;
  Eigen::MatrixXd k_;
};

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

struct MorseBond {
  int i;
  int j;
  double depth;
  double alpha;
  double r_eq;
};

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
[[nodiscard]] Eigen::VectorXd geometry_noise(const Eigen::VectorXd& coords, double amplitude,
                                             std::uint64_t seed);
}

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

}
