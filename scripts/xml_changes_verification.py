import mujoco
from gymnasium.utils.env_match import check_environments_match

import metaworld

SEED = 42
NUM_STEPS = 100_000

print("MuJoCo Version:", mujoco.__version__)

ENVS_TO_TEST = ["shelf-place-v2", "soccer-v2", "stick-push-v2", "stick-pull-v2"]

for env in ENVS_TO_TEST:
    print(f"Checking {env}...")
    try:
        env_name_1 = env
        env_name_2 = f"{env}-original"

        b1 = metaworld.MT1(env_name_1, seed=SEED)
        env_cls_name_1 = list(b1.train_classes.keys())[0]
        env_1 = b1.train_classes[env_cls_name_1]()
        task_1 = [task for task in b1.train_tasks if task.env_name == env_cls_name_1][0]
        env_1.set_task(task_1)

        b2 = metaworld.MT1(env_name_2, seed=SEED)
        env_cls_name_2 = list(b2.train_classes.keys())[0]
        env_2 = b2.train_classes[env_cls_name_2]()
        task_2 = [task for task in b2.train_tasks if task.env_name == env_cls_name_2][0]
        env_2.set_task(task_2)

        check_environments_match(
            env_1, env_2, num_steps=NUM_STEPS, seed=SEED, skip_render=True
        )
        print(f"Checked {env}.\n")
    except Exception as e:
        print(f"Check failed: {e}\n")
