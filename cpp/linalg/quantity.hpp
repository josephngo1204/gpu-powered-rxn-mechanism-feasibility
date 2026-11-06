#pragma once

#include <compare>

namespace saddle {

template <class Tag>
class Quantity {
 public:
  constexpr Quantity() = default;
  constexpr explicit Quantity(double value) noexcept : value_(value) {}

  [[nodiscard]] constexpr double value() const noexcept { return value_; }

  constexpr Quantity& operator+=(Quantity other) noexcept {
    value_ += other.value_;
    return *this;
  }
  constexpr Quantity& operator-=(Quantity other) noexcept {
    value_ -= other.value_;
    return *this;
  }

  friend constexpr Quantity operator+(Quantity a, Quantity b) noexcept {
    return Quantity{a.value_ + b.value_};
  }
  friend constexpr Quantity operator-(Quantity a, Quantity b) noexcept {
    return Quantity{a.value_ - b.value_};
  }
  friend constexpr Quantity operator-(Quantity a) noexcept { return Quantity{-a.value_}; }
  friend constexpr Quantity operator*(Quantity a, double s) noexcept {
    return Quantity{a.value_ * s};
  }
  friend constexpr Quantity operator*(double s, Quantity a) noexcept {
    return Quantity{a.value_ * s};
  }
  friend constexpr Quantity operator/(Quantity a, double s) noexcept {
    return Quantity{a.value_ / s};
  }
  friend constexpr auto operator<=>(Quantity, Quantity) noexcept = default;

 private:
  double value_ = 0.0;
};

struct BohrTag {};
struct HartreeTag {};
struct HartreePerBohrTag {};

using Bohr = Quantity<BohrTag>;
using Hartree = Quantity<HartreeTag>;
using HartreePerBohr = Quantity<HartreePerBohrTag>;

inline constexpr double kBohrToAngstrom = 0.529177210903;

[[nodiscard]] constexpr double to_angstrom(Bohr x) noexcept { return x.value() * kBohrToAngstrom; }
[[nodiscard]] constexpr Bohr bohr_from_angstrom(double angstrom) noexcept {
  return Bohr{angstrom / kBohrToAngstrom};
}

}
