from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation

from metaworld.asset_path_utils import full_V3_path_for
from metaworld.sawyer_xyz_env import RenderMode, SawyerXYZEnv
from metaworld.types import InitConfigDict

import metaworld_cpp.reward_utils as reward_utils_cpp


class SawyerSoccerEnvV3(SawyerXYZEnv):
    OBJ_RADIUS: float = 0.013
    TARGET_RADIUS: float = 0.07

    def __init__(
        self,
        render_mode: RenderMode | None = None,
        camera_name: str | None = None,
        camera_id: int | None = None,
        reward_function_version: str = "v2",
    ) -> None:
        goal_low = (-0.1, 0.8, 0.0)
        goal_high = (0.1, 0.9, 0.0)
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.6, 0.03)
        obj_high = (0.1, 0.7, 0.03)

        super().__init__(
            hand_low=hand_low,
            hand_high=hand_high,
            render_mode=render_mode,
            camera_name=camera_name,
            camera_id=camera_id,
        )
        self.reward_function_version = reward_function_version

        self.init_config: InitConfigDict = {
            "obj_init_pos": np.array([0, 0.6, 0.03]),
            "obj_init_angle": 0.3,
            "hand_init_pos": np.array([0.0, 0.6, 0.2]),
        }
        self.goal = np.array([0.0, 0.9, 0.03])
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.obj_init_angle = self.init_config["obj_init_angle"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
            dtype=np.float64,
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high), dtype=np.float64)

    @property
    def model_name(self) -> str:
        return full_V3_path_for("sawyer_xyz/sawyer_soccer.xml")

    @SawyerXYZEnv._Decorators.assert_task_is_set
    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
        obj = obs[4:7]
        (
            reward,
            tcp_to_obj,
            tcp_opened,
            target_to_obj,
            object_grasped,
            in_place,
        ) = self.compute_reward(action, obs)

        success = float(target_to_obj <= 0.07)
        near_object = float(tcp_to_obj <= 0.03)
        assert self.obj_init_pos is not None
        grasp_success = float(
            self.touching_main_object
            and (tcp_opened > 0)
            and (obj[2] - 0.02 > self.obj_init_pos[2])
        )
        info = {
            "success": success,
            "near_object": near_object,
            "grasp_success": grasp_success,
            "grasp_reward": object_grasped,
            "in_place_reward": in_place,
            "obj_to_target": target_to_obj,
            "unscaled_reward": reward,
        }

        return reward, info

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        return self.get_body_com("soccer_ball")

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        geom_xmat = self.data.body("soccer_ball").xmat.reshape(3, 3)
        return Rotation.from_matrix(geom_xmat).as_quat()

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()
        self._target_pos = self.goal.copy()
        self.obj_init_angle = self.init_config["obj_init_angle"]

        goal_pos = self._get_state_rand_vec()
        self._target_pos = goal_pos[3:]
        while np.linalg.norm(goal_pos[:2] - self._target_pos[:2]) < 0.15:
            goal_pos = self._get_state_rand_vec()
            self._target_pos = goal_pos[3:]
        assert self.obj_init_pos is not None
        self.obj_init_pos = np.concatenate([goal_pos[:2], [self.obj_init_pos[-1]]])
        self.model.body("goal_whole").pos = self._target_pos
        self._set_obj_xyz(self.obj_init_pos)
        self.maxPushDist = np.linalg.norm(
            self.obj_init_pos[:2] - np.array(self._target_pos)[:2]
        )

        self.model.site("goal").pos = self._target_pos

        return self._get_obs()

    def compute_reward(
        self, action: npt.NDArray[Any], obs: npt.NDArray[np.float64]
    ) -> tuple[float, float, float, float, float, float]:
        assert self._target_pos is not None and self.obj_init_pos is not None
        if self.reward_function_version == "v2":
            obj = obs[4:7]
            tcp_opened: float = obs[3]
            x_scaling = np.array([3.0, 1.0, 1.0])
            tcp_to_obj = float(np.linalg.norm(obj - self.tcp_center))
            target_to_obj = float(np.linalg.norm((obj - self._target_pos) * x_scaling))
            target_to_obj_init = float(
                np.linalg.norm((obj - self.obj_init_pos) * x_scaling)
            )

            in_place = reward_utils_cpp.tolerance(
                target_to_obj,
                bounds=(0, self.TARGET_RADIUS),
                margin=target_to_obj_init,
                sigmoid=reward_utils_cpp.SigmoidType.LongTail,
            )

            goal_line = self._target_pos[1] - 0.1
            if obj[1] > goal_line and abs(obj[0] - self._target_pos[0]) > 0.10:
                in_place = np.clip(
                    in_place - 2 * ((obj[1] - goal_line) / (1 - goal_line)), 0.0, 1.0
                )

            object_grasped = self._gripper_caging_reward(
                action,
                obj,
                self.OBJ_RADIUS,
                pad_success_thresh=0.05,
                xz_thresh=0.005,
                caging_thresh=0.95,
                grip_success_thresh=self.OBJ_RADIUS + 0.01,
                base_caging_thresh_on_obj_init_pos=False,
                use_abs_pad_to_obj_lr=False,
                compute_gripping_separately=True,
            )

            reward = (3 * object_grasped) + (6.5 * in_place)

            if target_to_obj < self.TARGET_RADIUS:
                reward = 10.0
            return (
                reward,
                tcp_to_obj,
                tcp_opened,
                float(np.linalg.norm(obj - self._target_pos)),
                object_grasped,
                in_place,
            )
