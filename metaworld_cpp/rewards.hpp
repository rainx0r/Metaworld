
#pragma once

#include "mujoco/mjtnum.h"

double gripper_caging_reward(
    const float *action, const mjtNum *obj_pos, const mjtNum *obj_init_pos,
    const mjtNum *left_pad, const mjtNum *right_pad, const mjtNum *tcp,
    const mjtNum *init_tcp, const mjtNum *init_left_pad,
    const mjtNum *init_right_pad, double obj_radius, double pad_success_thresh,
    double xz_thresh, double object_reach_radius = 0.01,
    double desired_gripper_effort = 1.0, double caging_thresh = 0.97,
    bool compute_gripping_separately = false,
    bool base_caging_thresh_on_obj_init_pos = true,
    double grip_success_thresh = 0.025, bool use_abs_pad_to_obj_lr = true,
    bool high_density = false, bool medium_density = false);
