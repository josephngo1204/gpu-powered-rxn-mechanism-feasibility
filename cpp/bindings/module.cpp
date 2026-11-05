// Python bindings. Model surfaces only for now; cuEST models arrive in Phase F.
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <tuple>
#include <vector>
#include <utility>

#include "core/model.hpp"
#include "hess/fd_hessian.hpp"
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
    } catch (const std::invalid_argument& e) {
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

  m.def(
      "central_stencil",
      [](const Eigen::VectorXd& coords, double step) {
        saddle::Geometry g;
        g.coords = coords;
        std::vector<std::tuple<long, int, Eigen::VectorXd>> out;
        for (auto& p : saddle::hess::central_stencil(g, step)) {
          out.emplace_back(static_cast<long>(p.coordinate), p.sign, std::move(p.geometry.coords));
        }
        return out;
      },
      "coords"_a, "step"_a,
      "Central-difference stencil as (coordinate, sign, displaced coords), in order.");

  m.def(
      "assemble_central",
      [](const std::vector<Eigen::VectorXd>& gradients, double step) {
        auto r = saddle::hess::assemble_central(gradients, step);
        return std::make_tuple(std::move(r.raw), std::move(r.hessian), r.asymmetry);
      },
      "gradients"_a, "step"_a,
      "Assemble (raw, symmetrized, relative asymmetry) from stencil-ordered gradients.");
}
