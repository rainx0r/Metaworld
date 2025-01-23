import mujoco._structs

def touching_object(model: mujoco._structs.MjModel, data: mujoco._structs.MjData, object_geom_id: int) -> bool:
    """Determines whether the gripper is touching the object with given id.

    Args:
        model(mujoco._structs.MjModel): the mujoco model
        data(mujoco._structs.MjData): the mujoco data
        object_geom_id(int): the ID of the object in question

    Returns:
        Whether the gripper is touching the object
    """
    ...
