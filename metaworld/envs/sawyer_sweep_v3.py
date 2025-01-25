from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box

from metaworld.asset_path_utils import full_V3_path_for
from metaworld.sawyer_xyz_env import RenderMode, SawyerXYZEnv
from metaworld.types import InitConfigDict

import metaworld_cpp.reward_utils as reward_utils_cpp


class SawyerSweepEnvV3(SawyerXYZEnv):
    OBJ_RADIUS: float = 0.02

    def __init__(
        self,
        render_mode: RenderMode | None = None,
        camera_name: str | None = None,
        camera_id: int | None = None,
        reward_function_version: str = "v2",
    ) -> None:
        init_puck_z = 0.1
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1.0, 0.5)
        obj_low = (-0.1, 0.6, 0.02)
        obj_high = (0.1, 0.7, 0.02)
        goal_low = (0.49, 0.6, 0.00)
        goal_high = (0.51, 0.7, 0.02)

        super().__init__(
            hand_low=hand_low,
            hand_high=hand_high,
            render_mode=render_mode,
            camera_name=camera_name,
            camera_id=camera_id,
        )
        self.reward_function_version = reward_function_version

        self.init_config: InitConfigDict = {
            "obj_init_pos": np.array([0.0, 0.6, 0.02]),
            "obj_init_angle": 0.3,
            "hand_init_pos": np.array([0.0, 0.6, 0.2]),
        }
        self.goal = np.array([0.5, 0.65, 0.01])
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.obj_init_angle = self.init_config["obj_init_angle"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        self.init_puck_z = init_puck_z

        self._random_reset_space = Box(
            np.array(obj_low), np.array(obj_high), dtype=np.float64
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high), dtype=np.float64)

    @property
    def model_name(self) -> str:
        return full_V3_path_for("sawyer_xyz/sawyer_sweep_v3.xml")

    @SawyerXYZEnv._Decorators.assert_task_is_set
    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
        (
            reward,
            tcp_to_obj,
            tcp_opened,
            target_to_obj,
            object_grasped,
            in_place,
        ) = self.compute_reward(action, obs)

        grasp_success = float(self.touching_main_object and (tcp_opened > 0))

        info = {
            "success": float(target_to_obj <= 0.05),
            "near_object": float(tcp_to_obj <= 0.03),
            "grasp_reward": object_grasped,
            "grasp_success": grasp_success,
            "in_place_reward": in_place,
            "obj_to_target": target_to_obj,
            "unscaled_reward": reward,
        }
        return reward, info

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        return self.data.body("obj").xquat

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        return self.data.body("obj").xpos

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()
        self._target_pos = self.goal.copy()
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.objHeight = self._get_pos_objects()[2]

        obj_pos = self._get_state_rand_vec()
        self.obj_init_pos = np.concatenate([obj_pos[:2], [self.obj_init_pos[-1]]])
        self._target_pos[1] = obj_pos.copy()[1]

        self._set_obj_xyz(self.obj_init_pos)
        self.model.site("goal").pos = self._target_pos
        return self._get_obs()

    def compute_reward(
        self, action: npt.NDArray[Any], obs: npt.NDArray[np.float64]
    ) -> tuple[float, float, float, float, float, float]:
        assert self._target_pos is not None
        if self.reward_function_version == "v2":
            _TARGET_RADIUS: float = 0.05
            tcp = self.tcp_center
            obj = obs[4:7]
            tcp_opened = obs[3]
            target = self._target_pos

            obj_to_target = float(np.linalg.norm(obj - target))
            tcp_to_obj = float(np.linalg.norm(obj - tcp))
            in_place_margin = np.linalg.norm(self.obj_init_pos - target)

            in_place = reward_utils_cpp.tolerance(
                obj_to_target,
                bounds=(0, _TARGET_RADIUS),
                margin=in_place_margin,
                sigmoid=reward_utils_cpp.SigmoidType.LongTail,
            )

            object_grasped = self._gripper_caging_reward(
                action,
                obj,
                self.OBJ_RADIUS,
                pad_success_thresh=0.05,
                grip_success_thresh=self.OBJ_RADIUS + 0.01,
                compute_gripping_separately=True,
                xz_thresh=0.005,
                use_abs_pad_to_obj_lr=False,
                base_caging_thresh_on_obj_init_pos=False,
                caging_thresh=0.95,
            )
            in_place_and_object_grasped = reward_utils_cpp.hamacher_product(
                object_grasped, in_place
            )

            reward = (2 * object_grasped) + (6 * in_place_and_object_grasped)

            if obj_to_target < _TARGET_RADIUS:
                reward = 10.0
            return (
                reward,
                tcp_to_obj,
                tcp_opened,
                obj_to_target,
                object_grasped,
                in_place,
            )
