#pragma once

#include <Eigen/Dense>
#include <concepts>
#include <initializer_list>
#include <vector>

#include "linalg/quantity.hpp"

namespace saddle {

struct Geometry {
  std::vector<int> atomic_numbers;
  Eigen::VectorXd coords;

  [[nodiscard]] static Geometry cartesian(std::initializer_list<double> values) {
    Geometry g;
    g.coords.resize(static_cast<Eigen::Index>(values.size()));
    Eigen::Index i = 0;
    for (double v : values) g.coords[i++] = v;
    return g;
  }

  [[nodiscard]] Eigen::Index dimension() const noexcept { return coords.size(); }
};

struct WarmStart {};

struct Diagnostics {
  bool converged = true;
  int scf_iterations = 0;
};

struct Evaluation {
  Hartree energy;
  Eigen::VectorXd gradient;
  Diagnostics diagnostics;
};

template <class Model>
concept ElectronicModel =
    requires(Model& model, const Geometry& geometry, const WarmStart& guess) {
      { model.evaluate(geometry, guess) } -> std::same_as<Evaluation>;
    };

}
