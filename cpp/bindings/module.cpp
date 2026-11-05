// Python bindings. Model surfaces only for now; cuEST models arrive in Phase F.
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>

#include <cstdint>
#include <utility>

#include "core/model.hpp"
#include "surfaces/surfaces.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

template <class Model>
std::pair<double, Eigen::VectorXd> evaluate(Model& model, const Eigen::VectorXd& coords) {
  saddle::Geometry g;
  g.coords = coords;
  saddle::Evaluation e = model.evaluate(g, {});
  return {e.energy.value(), std::move(e.gradient)};
}

constexpr const char* kEvaluateDoc =
    "Return (energy in Eh, gradient in Eh/bohr) at flattened coordinates in bohr.";

}  // namespace

NB_MODULE(_core, m) {
  m.doc() = "Saddle C++ core";

  nb::register_exception_translator([](const std::exception_ptr& p, void*) {
    try {
      std::rethrow_exception(p);
    } catch (const saddle::ShapeError& e) {
      PyErr_SetString(PyExc_ValueError, e.what());
    }
  });

  using saddle::surfaces::Harmonic;
  using saddle::surfaces::Morse;
  using saddle::surfaces::MullerBrown;
  using NoisyMullerBrown = saddle::surfaces::Noisy<MullerBrown>;

  nb::class_<MullerBrown>(m, "MullerBrown")
      .def(nb::init<>())
      .def("evaluate", &evaluate<MullerBrown>, "coords"_a, kEvaluateDoc);

  nb::class_<Morse>(m, "Morse")
      .def(nb::init<double, double, double>(), "depth"_a, "alpha"_a, "r_eq"_a)
      .def("evaluate", &evaluate<Morse>, "coords"_a, kEvaluateDoc);

  nb::class_<Harmonic>(m, "Harmonic")
      .def(nb::init<Eigen::VectorXd, Eigen::MatrixXd>(), "center"_a, "force_constants"_a)
      .def("evaluate", &evaluate<Harmonic>, "coords"_a, kEvaluateDoc);

  nb::class_<NoisyMullerBrown>(m, "NoisyMullerBrown")
      .def(
          "__init__",
          [](NoisyMullerBrown* self, double amplitude, std::uint64_t seed) {
            new (self) NoisyMullerBrown(MullerBrown{}, amplitude, seed);
          },
          "amplitude"_a, "seed"_a)
      .def("evaluate", &evaluate<NoisyMullerBrown>, "coords"_a, kEvaluateDoc);
}
