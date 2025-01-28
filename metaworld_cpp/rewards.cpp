#include "rewards.hpp"
#include "mujoco/mjtnum.h"
#include "reward_utils.hpp"
#include <cmath>
#include <stdexcept>

double gripper_caging_reward(
    const float *action, const mjtNum *obj_pos, const mjtNum *obj_init_pos,
    const mjtNum *left_pad, const mjtNum *right_pad, const mjtNum *tcp,
    const mjtNum *init_tcp, const mjtNum *init_left_pad,
    const mjtNum *init_right_pad, double obj_radius, double pad_success_thresh,
    double xz_thresh, double object_reach_radius, double desired_gripper_effort,
    double caging_thresh, bool compute_gripping_separately,
    bool base_caging_thresh_on_obj_init_pos, double grip_success_thresh,
    bool use_abs_pad_to_obj_lr, bool high_density, bool medium_density) {
  if (high_density && medium_density) {
    throw std::invalid_argument(
        "Can only be either high_density or medium_density");
  }

  // Get current positions of left and right pads (Y axis)
  double pad_y_left = left_pad[1], pad_y_right = right_pad[1];

  double pad_to_obj_left, pad_to_obj_right;
  if (use_abs_pad_to_obj_lr) {
    // Compare *current* pad positions with *current* obj position (Y axis)
    pad_to_obj_left = std::abs(pad_y_left - obj_pos[1]);
    pad_to_obj_right = std::abs(pad_y_right - obj_pos[1]);
  } else {
    // HACK: For soccer/pick-place/push-back/sweep/sweep-into
    pad_to_obj_left = left_pad[1] - obj_pos[1];
    pad_to_obj_right = obj_pos[1] - right_pad[1];
  }

  // Compute left/right caging rewards
  /* Compute the left/right caging rewards. This is crucial for success,
     yet counterintuitive mathematically because we invented it
     accidentally.
     Before touching the object, `pad_to_obj_lr` ("x") is always separated
     from `caging_lr_margin` ("the margin") by some small number,
     `pad_success_thresh`.

     When far away from the object:
         x = margin + pad_success_thresh
         --> Thus x is outside the margin, yielding very small reward.
             Here, any variation in the reward is due to the fact that
             the margin itself is shifting.
     When near the object (within pad_success_thresh):
         x = pad_success_thresh - margin
         --> Thus x is well within the margin. As long as x > obj_radius,
             it will also be within the bounds, yielding maximum reward.
             Here, any variation in the reward is due to the gripper
             moving *too close* to the object (i.e, blowing past the
             obj_radius bound).

     Therefore, before touching the object, this is very nearly a binary
     reward -- if the gripper is between obj_radius and pad_success_thresh,
     it gets maximum reward. Otherwise, the reward very quickly falls off.

     After grasping the object and moving it away from initial position,
     x remains (mostly) constant while the margin grows considerably. This
     penalizes the agent if it moves *back* toward `obj_init_pos`, but
     offers no encouragement for leaving that position in the first place.
     That part is left to the reward functions of individual environments. */
  double caging_left_margin, caging_right_margin;
  if (base_caging_thresh_on_obj_init_pos) {
    caging_left_margin =
        std::abs(std::abs(pad_y_left - obj_init_pos[1]) - pad_success_thresh);
    caging_right_margin =
        std::abs(std::abs(pad_y_right - obj_init_pos[1]) - pad_success_thresh);
  } else { // HACK: For soccer/pick-place/push-back/sweep/sweep-into
    caging_left_margin =
        std::abs(std::abs(init_left_pad[1] - obj_pos[1]) - pad_success_thresh);
    caging_right_margin =
        std::abs(std::abs(init_right_pad[1] - obj_pos[1]) - pad_success_thresh);
  }

  // TODO: replace pad_to_object_lr with pad_to_obj_left and pad_to_obj_right
  double caging_left = tolerance(
      pad_to_obj_left, std::make_tuple(obj_radius, pad_success_thresh),
      caging_left_margin, SigmoidType::LongTail);
  double caging_right = tolerance(
      pad_to_obj_right, std::make_tuple(obj_radius, pad_success_thresh),
      caging_right_margin, SigmoidType::LongTail);
  double caging_y = hamacher_product(caging_left, caging_right);

  double gripping_y;
  if (compute_gripping_separately) {
    // HACK: For socccer / push-back / sweep / sweep-into

    double gripping_left = tolerance(
        pad_to_obj_left, std::make_tuple(obj_radius, grip_success_thresh),
        caging_left_margin, SigmoidType::LongTail);
    double gripping_right = tolerance(
        pad_to_obj_right, std::make_tuple(obj_radius, grip_success_thresh),
        caging_right_margin, SigmoidType::LongTail);
    gripping_y = hamacher_product(gripping_left, gripping_right);
  }

  double caging_xz_margin =
      std::sqrt(std::pow(obj_init_pos[0] - init_tcp[0], 2) + // L2 Norm
                std::pow(obj_init_pos[2] - init_tcp[2], 2)) -
      xz_thresh;

  double tcp_to_obj_xz_norm =
      std::sqrt(std::pow(tcp[0] - obj_pos[0], 2) + // L2 Norm
                std::pow(tcp[2] - obj_pos[2], 2));

  double caging_xz =
      tolerance(tcp_to_obj_xz_norm, std::make_tuple(0.0, xz_thresh),
                caging_xz_margin, SigmoidType::LongTail);

  // Combine components
  double caging = hamacher_product(caging_y, caging_xz);
  double caging_and_gripping;

  if (compute_gripping_separately) {
    // HACK: For socccer / push-back / sweep / sweep-into
    double gripping = (caging > caging_thresh) ? gripping_y : 0.0;
    caging_and_gripping = (caging + gripping) / 2.0;
  } else {
    double gripper_closed =
        std::fmin(std::fmax(0.0, action[3]), desired_gripper_effort) /
        desired_gripper_effort;
    double gripping = (caging > caging_thresh) ? gripper_closed : 0.0;
    caging_and_gripping = hamacher_product(caging, gripping);
  }

  if (high_density) {
    caging_and_gripping = (caging_and_gripping + caging) / 2.0;
  }
  if (medium_density) {
    double tcp_to_obj = 0.0;
    double tcp_to_obj_init = 0.0;
    for (int i = 0; i < 3; i++) {
      tcp_to_obj += std::pow(obj_pos[i] - tcp[i], 2);
      tcp_to_obj_init += std::pow(obj_init_pos[i] - init_tcp[i], 2);
    }
    tcp_to_obj = std::sqrt(tcp_to_obj);
    tcp_to_obj_init = std::sqrt(tcp_to_obj_init);

    double reach_margin = std::abs(tcp_to_obj_init - object_reach_radius);
    double reach =
        tolerance(tcp_to_obj, std::make_tuple(0.0, object_reach_radius),
                  reach_margin, SigmoidType::LongTail);
    caging_and_gripping = (caging_and_gripping + reach) / 2.0;
  }
  return caging_and_gripping;
}
