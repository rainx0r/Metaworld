import pathlib
import sys

import gymnasium.utils
import gymnasium.utils.save_video
import mujoco
import polars as pl
from tqdm import tqdm  # type: ignore

import metaworld


def main() -> None:
    SEED = 42
    NUM_STEPS = 1_000

    if not len(sys.argv) > 1:
        raise ValueError("Please provide the task name.")
    if sys.argv[1] not in metaworld.MT1.ENV_NAMES:
        raise ValueError(
            f'Task "{sys.argv[1]}" not found, valid options: {metaworld.MT1.ENV_NAMES}'
        )
    env_name = sys.argv[1]

    if len(sys.argv) > 2:
        try:
            NUM_STEPS = int(sys.argv[2])
        except ValueError:
            raise ValueError(
                f"Invalid number of steps {sys.argv[2]}. Please use a valid integer."
            )

    print("MuJoCo version:", mujoco.__version__)
    print("Number of steps:", NUM_STEPS)
    print("Env:", env_name)
    print("Seed:", SEED)

    observations = []
    frames = []
    actions = []
    rewards = []
    terminateds = []
    truncateds = []

    b = metaworld.MT1(env_name, seed=SEED)
    env_cls_name = list(b.train_classes.keys())[0]
    env = b.train_classes[env_cls_name](camera_name="corner3", render_mode="rgb_array")
    task = [task for task in b.train_tasks if task.env_name == env_cls_name][0]
    env.set_task(task)
    env.reset(seed=SEED)
    env.action_space.seed(seed=SEED)
    env.max_path_length = NUM_STEPS

    for _ in tqdm(range(NUM_STEPS)):
        action = env.action_space.sample()
        obs, rew, terminated, truncated, _ = env.step(action)
        frame = env.render().copy()

        observations.append(obs)
        actions.append(action)
        frames.append(frame)
        rewards.append(rew)
        terminateds.append(terminated)
        truncateds.append(truncated)

    # Save video
    videos_dir = pathlib.Path("videos")
    videos_dir.mkdir(exist_ok=True)
    gymnasium.utils.save_video.save_video(
        frames, str(videos_dir), name_prefix=env_name, fps=env.metadata["render_fps"]
    )

    # Save trajectory to parquet
    data_dir = pathlib.Path("trajectories")
    data_dir.mkdir(exist_ok=True)
    df = pl.DataFrame(
        {
            "obs": observations,
            "action": actions,
            "reward": rewards,
            "terminated": terminateds,
            "truncated": truncateds,
        }
    )
    df.write_parquet(data_dir / f"{env_name}.parquet")


if __name__ == "__main__":
    main()
