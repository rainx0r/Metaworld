#include <mujoco/mujoco.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "mujoco_utils.cpp"
#include "reward_utils.cpp"

namespace py = pybind11;
PYBIND11_MODULE(metaworld_cpp, m) {
  auto reward_utils =
      m.def_submodule("reward_utils", "Reward utility functions");
  reward_utils.def("hamacher_product", &hamacher_product,
                   "Returns the hamacher (t-norm) product of a and b");
  reward_utils.def(
      "rect_prism_tolerance",
      [](py::array_t<double, py::array::c_style> curr,
         py::array_t<double, py::array::c_style> zero,
         py::array_t<double, py::array::c_style> one) {
        return rect_prism_tolerance(curr.data(), zero.data(), one.data());
      },
      "Computes a reward if curr is inside a rectangular prism region. All "
      "inputs are 3D points with shape (3,).",
      py::arg("curr"), py::arg("zero"), py::arg("one"));

  py::enum_<SigmoidType>(reward_utils, "SigmoidType")
      .value("Gaussian", SigmoidType::Gaussian)
      .value("Hyperbolic", SigmoidType::Hyperbolic)
      .value("LongTail", SigmoidType::LongTail)
      .value("Reciprocal", SigmoidType::Reciprocal)
      .value("Cosine", SigmoidType::Cosine)
      .value("Linear", SigmoidType::Linear)
      .value("Quadratic", SigmoidType::Quadratic)
      .value("TanhSquared", SigmoidType::TanhSquared)
      .export_values();

  reward_utils.def("tolerance", &tolerance,
                   "Returns 1 when x falls inside the bounds, between 0 and 1 "
                   "otherwise.",
                   py::arg("x"), py::arg("bounds") = std::make_tuple(0.0, 0.0),
                   py::arg("margin") = 0.0,
                   py::arg("sigmoid") = SigmoidType::Gaussian);

  m.def(
      "touching_object",
      [](py::object m, py::object d, int object_geom_id) {
        std::uintptr_t d_raw = d.attr("_address").cast<std::uintptr_t>();
        std::uintptr_t m_raw = m.attr("_address").cast<std::uintptr_t>();
        mjData *d_cpp_ = reinterpret_cast<mjData *>(d_raw);
        mjModel *m_cpp_ = reinterpret_cast<mjModel *>(m_raw);
        return touching_object(m_cpp_, d_cpp_, object_geom_id);
      },
      "Determines whether the gripper is touching the object with given id.");
}
