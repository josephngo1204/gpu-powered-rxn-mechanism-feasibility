#pragma once

// Physical constants (CODATA 2018). The single C++ source of truth; the
// Python side mirrors these values in python/saddle/units.py.
namespace saddle::constants {

inline constexpr double kHartreeToWavenumber = 219474.6313632;  // cm^-1 per hartree
inline constexpr double kAmuToElectronMass = 1822.888486209;     // electron masses per dalton

}  // namespace saddle::constants
