"""Sets up the project."""

import pathlib

import mujoco
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "metaworld_cpp",
        [
            "metaworld_cpp/bindings.cpp",
            "metaworld_cpp/mujoco_utils.cpp",
            "metaworld_cpp/reward_utils.cpp",
            "metaworld_cpp/rewards.cpp",
        ],
        include_dirs=[pathlib.Path(mujoco.HEADERS_DIR).parent],
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
