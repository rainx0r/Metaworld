#include <mujoco/mujoco.h>
#include <pybind11/pybind11.h>

#include "mujoco_utils.cpp"

namespace py = pybind11;

PYBIND11_MODULE(metaworld_cpp, m) {
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
