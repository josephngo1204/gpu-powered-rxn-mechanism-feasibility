#pragma once

namespace saddle {

struct Restricted {};
struct Unrestricted {};

template <class Spin>
struct SpinTraits;

// J/K gradient scale factors depend on the spin formulation (spec §8).
template <>
struct SpinTraits<Restricted> {
  static constexpr double coulomb_gradient_scale = 2.0;
  static constexpr double exchange_gradient_scale = -1.0;
  static constexpr int n_density_matrices = 1;
};

template <>
struct SpinTraits<Unrestricted> {
  static constexpr double coulomb_gradient_scale = 0.5;
  static constexpr double exchange_gradient_scale = -0.5;
  static constexpr int n_density_matrices = 2;
};

}  // namespace saddle
