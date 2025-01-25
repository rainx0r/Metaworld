from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation

from metaworld.asset_path_utils import full_V3_path_for
from metaworld.sawyer_xyz_env import RenderMode, SawyerXYZEnv
from metaworld.types import ObservationDict, StickInitConfigDict

import metaworld_cpp.reward_utils as reward_utils_cpp


class SawyerStickPushEnvV3(SawyerXYZEnv):
    def __init__(
        self,
        render_mode: RenderMode | None = None,
        camera_name: str | None = None,
        camera_id: int | None = None,
        reward_function_version: str = "v2",
    ) -> None:
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.08, 0.58, 0.000)
        obj_high = (-0.03, 0.62, 0.001)
        goal_low = (0.399, 0.55, 0.1319)
        goal_high = (0.401, 0.6, 0.1321)

        super().__init__(
            hand_low=hand_low,
            hand_high=hand_high,
            render_mode=render_mode,
            camera_name=camera_name,
            camera_id=camera_id,
        )
        self.reward_function_version = reward_function_version

        self.init_config: StickInitConfigDict = {
            "stick_init_pos": np.array([-0.1, 0.6, 0.02]),
            "hand_init_pos": np.array([0, 0.6, 0.2]),
        }
        self.goal = self.init_config["stick_init_pos"]
        self.stick_init_pos = self.init_config["stick_init_pos"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        # For now, fix the object initial position.
        self.obj_init_pos = np.array([0.2, 0.6, 0.0])
        self.obj_init_qpos = np.array([0.0, 0.0])
        self.obj_space = Box(np.array(obj_low), np.array(obj_high), dtype=np.float64)
        self.goal_space = Box(np.array(goal_low), np.array(goal_high), dtype=np.float64)
        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
            dtype=np.float64,
        )

    @property
    def model_name(self) -> str:
        return full_V3_path_for("sawyer_xyz/sawyer_stick_obj.xml")

    @SawyerXYZEnv._Decorators.assert_task_is_set
    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
        stick = obs[4:7]
        container = obs[11:14]
        (
            reward,
            tcp_to_obj,
            tcp_open,
            container_to_target,
            grasp_reward,
            stick_in_place,
        ) = self.compute_reward(action, obs)
        assert self._target_pos is not None
        success = float(np.linalg.norm(container - self._target_pos) <= 0.12)
        near_object = float(tcp_to_obj <= 0.03)
        grasp_success = float(
            self.touching_main_object
            and (tcp_open > 0)
            and (stick[2] - 0.01 > self.stick_init_pos[2])
        )

        info = {
            "success": grasp_success and success,
            "near_object": near_object,
            "grasp_success": grasp_success,
            "grasp_reward": grasp_reward,
            "in_place_reward": stick_in_place,
            "obj_to_target": container_to_target,
            "unscaled_reward": reward,
        }

        return reward, info

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        return np.hstack(
            (
                self.get_body_com("stick").copy(),
                self._get_site_pos("insertion") + np.array([0.0, 0.09, 0.0]),
            )
        )

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        geom_xmat = self.data.body("stick").xmat.reshape(3, 3)
        return np.hstack(
            (
                Rotation.from_matrix(geom_xmat).as_quat(),
                np.array(
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ]
                ),
            )
        )

    def _get_obs_dict(self) -> ObservationDict:
        obs_dict = super()._get_obs_dict()
        obs_dict["state_achieved_goal"] = self._get_site_pos("insertion") + np.array(
            [0.0, 0.09, 0.0]
        )
        return obs_dict

    def _set_stick_xyz(self, pos: npt.NDArray[Any]) -> None:
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9:12] = pos.copy()
        qvel[9:15] = 0
        self.set_state(qpos, qvel)

    def _set_obj_xyz(self, pos: npt.NDArray[Any]) -> None:
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[16:18] = pos.copy()
        qvel[16:18] = 0
        self.set_state(qpos, qvel)

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()
        self.stick_init_pos = self.init_config["stick_init_pos"]
        self._target_pos = np.array([0.4, 0.6, self.stick_init_pos[-1]])

        goal_pos = self._get_state_rand_vec()
        while np.linalg.norm(goal_pos[:2] - goal_pos[-3:-1]) < 0.1:
            goal_pos = self._get_state_rand_vec()
        self.stick_init_pos = np.concatenate([goal_pos[:2], [self.stick_init_pos[-1]]])
        self._target_pos = np.concatenate(
            [goal_pos[-3:-1], [self._get_site_pos("insertion")[-1]]]
        )

        self._set_stick_xyz(self.stick_init_pos)
        self._set_obj_xyz(self.obj_init_qpos)
        self.obj_init_pos = self.get_body_com("object").copy()

        self.model.site("goal").pos = self._target_pos

        return self._get_obs()

    def compute_reward(
        self, action: npt.NDArray[Any], obs: npt.NDArray[np.float64]
    ) -> tuple[float, float, float, float, float, float]:
        assert self._target_pos is not None and self.obj_init_pos is not None
        if self.reward_function_version == "v2":
            _TARGET_RADIUS: float = 0.12
            tcp = self.tcp_center
            stick = obs[4:7] + np.array([0.015, 0.0, 0.0])
            container = obs[11:14]
            tcp_opened: float = obs[3]
            target = self._target_pos

            tcp_to_stick = float(np.linalg.norm(stick - tcp))
            stick_to_target = float(np.linalg.norm(stick - target))
            stick_in_place_margin = float(
                np.linalg.norm(self.stick_init_pos - target) - _TARGET_RADIUS
            )
            stick_in_place = reward_utils_cpp.tolerance(
                stick_to_target,
                bounds=(0, _TARGET_RADIUS),
                margin=stick_in_place_margin,
                sigmoid=reward_utils_cpp.SigmoidType.LongTail,
            )

            container_to_target = float(np.linalg.norm(container - target))
            container_in_place_margin = float(
                np.linalg.norm(self.obj_init_pos - target) - _TARGET_RADIUS
            )
            container_in_place = reward_utils_cpp.tolerance(
                container_to_target,
                bounds=(0, _TARGET_RADIUS),
                margin=container_in_place_margin,
                sigmoid=reward_utils_cpp.SigmoidType.LongTail,
            )

            object_grasped = self._gripper_caging_reward(
                action=action,
                obj_pos=stick,
                obj_radius=0.04,
                pad_success_thresh=0.05,
                object_reach_radius=0.01,
                xz_thresh=0.01,
                obj_init_pos=self.stick_init_pos,
                high_density=True,
            )

            reward = object_grasped

            if (
                tcp_to_stick < 0.02
                and (tcp_opened > 0)
                and (stick[2] - 0.01 > self.stick_init_pos[2])
            ):
                object_grasped = 1
                reward = 2.0 + 5.0 * stick_in_place + 3.0 * container_in_place

                if container_to_target <= _TARGET_RADIUS:
                    reward = 10.0
            return (
                reward,
                tcp_to_stick,
                tcp_opened,
                container_to_target,
                object_grasped,
                stick_in_place,
            )
