// Python bindings. Model surfaces only for now; cuEST models arrive in Phase F.
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <tuple>
#include <vector>
#include <utility>

#include "core/model.hpp"
#include "hess/fd_hessian.hpp"
#include "hess/noise.hpp"
#include "hess/vibrations.hpp"
#include "search/irc.hpp"
#include "search/optimizer.hpp"
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

// Adapts any Python object with evaluate(coords) -> (energy, gradient) to the
// ElectronicModel concept, so C++ search code runs on Python-side models.
struct CallbackModel {
  nb::object target;
  saddle::Evaluation evaluate(const saddle::Geometry& g, const saddle::WarmStart& /*guess*/) {
    nb::object out = target.attr("evaluate")(g.coords);
    auto [energy, gradient] = nb::cast<std::pair<double, Eigen::VectorXd>>(out);
    if (gradient.size() != g.coords.size()) {
      throw saddle::ShapeError("model gradient length does not match coordinates");
    }
    return saddle::Evaluation{saddle::Hartree{energy}, std::move(gradient), {}};
  }
};
static_assert(saddle::ElectronicModel<CallbackModel>);

saddle::Geometry make_geometry(const Eigen::VectorXd& coords, const std::vector<int>& atomic_numbers) {
  saddle::Geometry g;
  g.coords = coords;
  g.atomic_numbers = atomic_numbers;
  return g;
}

nb::dict to_dict(const saddle::search::OptimizationResult& r) {
  nb::dict d;
  d["coords"] = r.geometry.coords;
  d["energy"] = r.evaluation.energy.value();
  d["gradient"] = r.evaluation.gradient;
  d["hessian"] = r.hessian;
  d["iterations"] = r.iterations;
  d["converged"] = r.converged;
  d["n_gradients"] = r.n_gradients;
  d["n_hessians"] = r.n_hessians;
  d["energies"] = r.energies;
  return d;
}

saddle::search::OptimizerOptions optimizer_options(int max_iterations, double trust_radius,
                                                   bool exact_hessian, double fd_step) {
  saddle::search::OptimizerOptions o;
  o.max_iterations = max_iterations;
  o.trust_radius = trust_radius;
  o.initial_hessian = exact_hessian ? saddle::search::InitialHessian::Exact
                                    : saddle::search::InitialHessian::Identity;
  o.fd_step = fd_step;
  return o;
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
    } catch (const std::domain_error& e) {
      PyErr_SetString(PyExc_ValueError, e.what());
    }
  });

  using saddle::surfaces::Harmonic;
  using saddle::surfaces::Morse;
  using saddle::surfaces::MullerBrown;
  using NoisyMullerBrown = saddle::surfaces::Noisy<MullerBrown>;
  using saddle::surfaces::Spring;
  using saddle::surfaces::Springs;

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

  nb::class_<Springs>(m, "Springs")
      .def(
          "__init__",
          [](Springs* self, const std::vector<std::tuple<int, int, double, double>>& springs) {
            std::vector<Spring> s;
            for (const auto& [i, j, k, r0] : springs) s.push_back({i, j, k, r0});
            new (self) Springs(std::move(s));
          },
          "springs"_a, "Pairwise harmonic springs given as (i, j, k, r0) tuples.")
      .def("evaluate", &evaluate<Springs>, "coords"_a, kEvaluateDoc);

  nb::class_<saddle::hess::VibrationalAnalysis>(m, "VibrationalAnalysis")
      .def_ro("frequencies_cm", &saddle::hess::VibrationalAnalysis::frequencies_cm)
      .def_ro("modes_mass_weighted", &saddle::hess::VibrationalAnalysis::modes_mass_weighted)
      .def_ro("modes_cartesian", &saddle::hess::VibrationalAnalysis::modes_cartesian)
      .def_ro("n_external", &saddle::hess::VibrationalAnalysis::n_external)
      .def_ro("linear", &saddle::hess::VibrationalAnalysis::linear)
      .def_ro("max_external_residual_cm",
              &saddle::hess::VibrationalAnalysis::max_external_residual_cm)
      .def_ro("translation_sum_rule", &saddle::hess::VibrationalAnalysis::translation_sum_rule)
      .def("n_imaginary", &saddle::hess::VibrationalAnalysis::n_imaginary, "threshold_cm"_a);

  using saddle::surfaces::MorseBond;
  using saddle::surfaces::PairModel;
  nb::class_<PairModel>(m, "PairModel")
      .def(
          "__init__",
          [](PairModel* self, const std::vector<std::tuple<int, int, double, double>>& springs,
             const std::vector<std::tuple<int, int, double, double, double>>& morse) {
            std::vector<Spring> s;
            for (const auto& [i, j, k, r0] : springs) s.push_back({i, j, k, r0});
            std::vector<MorseBond> mb;
            for (const auto& [i, j, dd, a, re] : morse) mb.push_back({i, j, dd, a, re});
            new (self) PairModel(std::move(s), std::move(mb));
          },
          "springs"_a, "morse"_a,
          "Springs (i, j, k, r0) plus Morse bonds (i, j, depth, alpha, r_eq).")
      .def("evaluate", &evaluate<PairModel>, "coords"_a, kEvaluateDoc);

  m.def(
      "minimize",
      [](nb::object model, const Eigen::VectorXd& coords, const std::vector<int>& atomic_numbers,
         int max_iterations, double trust_radius, bool exact_hessian, double fd_step) {
        CallbackModel cb{std::move(model)};
        return to_dict(saddle::search::minimize(
            cb, make_geometry(coords, atomic_numbers),
            optimizer_options(max_iterations, trust_radius, exact_hessian, fd_step)));
      },
      "model"_a, "coords"_a, "atomic_numbers"_a, "max_iterations"_a = 200, "trust_radius"_a = 0.1,
      "exact_hessian"_a = true, "fd_step"_a = 1e-4,
      "RS-RFO minimization to the G1 thresholds. Returns a dict of results.");

  m.def(
      "saddle_search",
      [](nb::object model, const Eigen::VectorXd& coords, const std::vector<int>& atomic_numbers,
         std::optional<Eigen::VectorXd> follow_direction, int max_iterations, double trust_radius,
         double fd_step) {
        CallbackModel cb{std::move(model)};
        saddle::search::SaddleOptions o;
        static_cast<saddle::search::OptimizerOptions&>(o) =
            optimizer_options(max_iterations, trust_radius, true, fd_step);
        if (follow_direction) o.follow_direction = *follow_direction;
        const auto r = saddle::search::saddle_search(cb, make_geometry(coords, atomic_numbers), o);
        nb::dict d = to_dict(r);
        d["n_recomputes"] = r.n_recomputes;
        d["followed_mode"] = r.followed_mode;
        d["mode_overlaps"] = r.mode_overlaps;
        return d;
      },
      "model"_a, "coords"_a, "atomic_numbers"_a, "follow_direction"_a = nb::none(),
      "max_iterations"_a = 200, "trust_radius"_a = 0.1, "fd_step"_a = 1e-4,
      "RS-P-RFO first-order saddle search with Bofill updates.");

  m.def(
      "irc",
      [](nb::object model, const Eigen::VectorXd& coords, const std::vector<int>& atomic_numbers,
         const Eigen::MatrixXd& hessian, const Eigen::VectorXd& masses, double step, int max_steps,
         double switch_gradient) {
        CallbackModel cb{std::move(model)};
        saddle::search::IrcOptions o;
        o.step = step;
        o.max_steps = max_steps;
        o.switch_gradient = switch_gradient;
        const auto r = saddle::search::irc(cb, make_geometry(coords, atomic_numbers), hessian, masses, o);
        auto branch = [](const saddle::search::IrcBranch& b) {
          nb::dict d;
          std::vector<Eigen::VectorXd> path;
          for (const auto& g : b.path) path.push_back(g.coords);
          d["path"] = path;
          d["energies"] = b.energies;
          d["endpoint"] = to_dict(b.endpoint);
          d["stop_reason"] = b.stop_reason;
          return d;
        };
        nb::dict d;
        d["forward"] = branch(r.forward);
        d["reverse"] = branch(r.reverse);
        d["transition_mode"] = r.transition_mode;
        d["transition_eigenvalue"] = r.transition_eigenvalue;
        return d;
      },
      "model"_a, "coords"_a, "atomic_numbers"_a, "hessian"_a, "masses"_a, "step"_a = 0.05,
      "max_steps"_a = 300, "switch_gradient"_a = 1e-3,
      "IRC in both directions from a first-order saddle, with endpoint optimization.");

  m.def(
      "probe_noise",
      [](nb::object model, const Eigen::VectorXd& coords, const Eigen::VectorXd& direction,
         double delta, int order, int n_samples) {
        CallbackModel cb{std::move(model)};
        const auto est = saddle::hess::probe_noise(cb, make_geometry(coords, {}), direction, delta,
                                                   order, n_samples);
        return std::make_tuple(est.epsilon, est.n_gradients);
      },
      "model"_a, "coords"_a, "direction"_a, "delta"_a = 1e-3, "order"_a = 6, "n_samples"_a = 16,
      "Line-scan gradient noise estimate; returns (epsilon Eh/bohr, gradients used).");

  m.def("optimal_step", &saddle::hess::optimal_step, "epsilon"_a, "fourth_derivative"_a,
        "Optimal central-difference step (3 eps / |fourth derivative|)^(1/3).");

  m.def("analyze_vibrations", &saddle::hess::analyze_vibrations, "hessian"_a, "coords"_a,
        "masses_amu"_a,
        "Harmonic analysis with translation/rotation projection. Frequencies in cm^-1, "
        "imaginary modes negative.");

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
