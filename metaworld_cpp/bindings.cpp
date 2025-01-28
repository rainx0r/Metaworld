#include <mujoco/mujoco.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "mujoco_utils.hpp"
#include "reward_utils.hpp"
#include "rewards.hpp"

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

  auto rewards = m.def_submodule("rewards", "Reward function components");
  rewards.def(
      "gripper_caging_reward",
      [](const py::array_t<float, py::array::c_style> action,
         const py::array_t<double, py::array::c_style> obj_pos,
         const py::array_t<double, py::array::c_style> obj_init_pos,
         const py::array_t<double, py::array::c_style> left_pad,
         const py::array_t<double, py::array::c_style> right_pad,
         const py::array_t<double, py::array::c_style> tcp,
         const py::array_t<double, py::array::c_style> init_tcp,
         const py::array_t<double, py::array::c_style> init_left_pad,
         const py::array_t<double, py::array::c_style> init_right_pad,
         double obj_radius, double pad_success_thresh, double xz_thresh,
         double object_reach_radius, double desired_gripper_effort,
         double caging_thresh, bool compute_gripping_separately,
         bool base_caging_thresh_on_obj_init_pos, double grip_success_thresh,
         bool use_abs_pad_to_obj_lr, bool high_density, bool medium_density) {
        return gripper_caging_reward(
            action.data(), obj_pos.data(), obj_init_pos.data(), left_pad.data(),
            right_pad.data(), tcp.data(), init_tcp.data(), init_left_pad.data(),
            init_right_pad.data(), obj_radius, pad_success_thresh, xz_thresh,
            object_reach_radius, desired_gripper_effort, caging_thresh,
            compute_gripping_separately, base_caging_thresh_on_obj_init_pos,
            grip_success_thresh, use_abs_pad_to_obj_lr, high_density,
            medium_density);
      },
      "Reward for agent grasping the object.", py::arg("action"),
      py::arg("obj_pos"), py::arg("obj_init_pos"), py::arg("left_pad"),
      py::arg("right_pad"), py::arg("tcp"), py::arg("init_tcp"),
      py::arg("init_left_pad"), py::arg("init_right_pad"),
      py::arg("obj_radius"), py::arg("pad_success_thresh"),
      py::arg("xz_thresh"), py::arg("object_reach_radius") = 0.01,
      py::arg("desired_gripper_effort") = 1.0, py::arg("caging_thresh") = 0.97,
      py::arg("compute_gripping_separately") = false,
      py::arg("base_caging_thresh_on_obj_init_pos") = true,
      py::arg("grip_success_thresh") = 0.025,
      py::arg("use_abs_pad_to_obj_lr") = true, py::arg("high_density") = false,
      py::arg("medium_density") = false);

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
