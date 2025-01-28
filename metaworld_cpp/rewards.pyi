import numpy as np
import numpy.typing as npt

def gripper_caging_reward(
    action: npt.NDArray[np.float32],
    obj_pos: npt.NDArray[np.float64],
    obj_init_pos: npt.NDArray[np.float64],
    left_pad: npt.NDArray[np.float64],
    right_pad: npt.NDArray[np.float64],
    tcp: npt.NDArray[np.float64],
    init_tcp: npt.NDArray[np.float64],
    init_left_pad: npt.NDArray[np.float64],
    init_right_pad: npt.NDArray[np.float64],
    obj_radius: float,
    pad_success_thresh: float,
    xz_thresh: float,
    object_reach_radius: float = 0.01,
    desired_gripper_effort: float = 1.0,
    caging_thresh: float = 0.97,
    compute_gripping_separately: bool = False,
    base_caging_thresh_on_obj_init_pos: bool = True,
    grip_success_thresh: float = 0.025,
    use_abs_pad_to_obj_lr: bool = True,
    high_density: bool = False,
    medium_density: bool = False,
) -> float:
    """Reward the agent for grasping the object.

    Args:
        action(np.ndarray): (4,) array representing the action
            delta(x), delta(y), delta(z), gripper_effort
        obj_pos(np.ndarray): (3,) array representing the obj x,y,z
        obj_init_pos(np.ndarray): (3,) array representing the initial object position. Only needed
            if the object the arm must interact with is not the same as `self.obj_init_pos`.
        left_pad(np.ndarray): (3,) array representing the left pad x,y,z
        right_pad(np.ndarray): (3,) array representing the right pad x,y,z
        tcp(np.ndarray): (3,) array representing the tcp x,y,z
        init_tcp(np.ndarray): (3,) array representing the initial tcp x,y,z
        init_left_pad(np.ndarray): (3,) array representing the initial left pad x,y,z
        init_right_pad(np.ndarray): (3,) array representing the initial right pad x,y,z
        obj_radius(float): radius of object's bounding sphere
        pad_success_thresh(float): successful distance of gripper_pad
            to object
        xz_thresh(float): successful distance of gripper in x_z axis to the
            object. Y axis not included since the caging function handles
                successful grasping in the Y axis.
        desired_gripper_effort(float): desired gripper effort, defaults to 1.0.
        caging_thresh(float): threshold for caging to give gripping reward, defaults to 0.97.
        compute_gripping_separately(bool): whether to compute gripping reward separately from the caging reward.
            Only needed for some reward functions.
        base_caging_thresh_on_obj_init_pos(bool): whether to use the base caging threshold on the initial object position (vs the initial pad position).
        grip_success_thresh(float): threshold for gripping to give gripping reward, only needed if `compute_gripping_separately` is True.
        use_abs_pad_to_obj_lr(bool): whether to use absolute pad-to-object distances in the caging reward.
        high_density(bool): flag for high-density. Cannot be used with medium-density.
        medium_density(bool): flag for medium-density. Cannot be used with high-density.

    Returns:
        the reward value
    """
