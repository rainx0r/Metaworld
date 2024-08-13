import pathlib
import sys

import carbs  # type: ignore
import compare_trajectories
import numpy as np
import polars as pl
import trajectory_collection
import wandb
import wandb.util

INITIAL_TORQUESCALE_VAL = 5.0
TARGET_METRIC = "loss/obs_mse"
TRIALS = 100
SEED = 42
NUM_STEPS = 1_000


def test_weld_settings(
    env_name: str, tgt_data_path: pathlib.Path, torquescale: float
) -> dict[str, np.float64]:
    tgt_data = pl.read_parquet(tgt_data_path)
    obs, _, _, rewards, _, _ = trajectory_collection.collect_trajectory(
        env_name=env_name, num_steps=NUM_STEPS, seed=SEED, torquscale=torquescale
    )

    obs_arr = np.array(obs)
    tgt_obs_arr = tgt_data["obs"].to_numpy()

    rew_arr = np.array(rewards)
    tgt_rew_arr = tgt_data["rew"].to_numpy()

    obs_mse, obs_mae = (
        compare_trajectories.get_mse_loss(obs_arr, tgt_obs_arr),
        compare_trajectories.get_mae_loss(obs_arr, tgt_obs_arr),
    )
    rew_mse, rew_mae = (
        compare_trajectories.get_mse_loss(rew_arr, tgt_rew_arr),
        compare_trajectories.get_mae_loss(rew_arr, tgt_rew_arr),
    )

    return {
        "loss/obs_mse": obs_mse,
        "loss/obs_mae": obs_mae,
        "loss/rew_mse": rew_mse,
        "loss/rew_mae": rew_mae,
    }


# From https://github.com/PufferAI/PufferLib/blob/carbs/demo.py#L136
def main() -> None:
    if len(sys.argv) < 3:
        raise ValueError("You need to provide an env name and path to the target data.")
    else:
        env_name = sys.argv[1]
        tgt_data_path = sys.argv[2]

    params = {"torquescale": {"distribution": "uniform", "min": 3.75, "max": 10.0}}
    sweep_id = wandb.sweep(
        sweep={
            "method": "bayes",
            "name": "sweep-obs",
            "metric": {
                "goal": "minimize",
                "name": TARGET_METRIC,
            },
            "parameters": params,
        },
        project="mw-weld-sweep",
    )
    wandb_params = params

    def carbs_param(
        name: str,
        space: str,
        min: float | None = None,
        max: float | None = None,
        search_center: float | None = None,
        is_integer: bool = False,
        rounding_factor: int = 1,
    ):
        wandb_param = wandb_params[name]
        if min is None:
            min = float(wandb_param["min"])  # type: ignore
        if max is None:
            max = float(wandb_param["max"])  # type: ignore

        if space == "log":
            Space = carbs.LogSpace
            if search_center is None:
                search_center = 2 ** (np.log2(min) + np.log2(max) / 2)
        elif space == "linear":
            Space = carbs.LinearSpace
            if search_center is None:
                search_center = (min + max) / 2
        elif space == "logit":
            Space = carbs.LogitSpace
            assert min == 0
            assert max == 1
            assert search_center is not None
        else:
            raise ValueError(f"Invalid CARBS space: {space} (log/linear)")

        assert search_center is not None

        return carbs.Param(
            name=name,
            space=Space(
                min=min, max=max, is_integer=is_integer, rounding_factor=rounding_factor
            ),
            search_center=search_center,
        )

    param_spaces = [
        carbs_param("torquescale", "linear", search_center=INITIAL_TORQUESCALE_VAL)
    ]
    carbs_params = carbs.CARBSParams(
        better_direction_sign=-1,
        is_wandb_logging_enabled=False,
        resample_frequency=0,
    )
    carbs_run = carbs.CARBS(carbs_params, param_spaces)

    def _sweep_main():
        wandb.init(
            id=wandb.util.generate_id(),
            project="mw-weld-sweep",
            entity="evangelos-ch",
            group="mw-dev",
            name="mw-mj3-weld",
            config={"torquescale": INITIAL_TORQUESCALE_VAL},
        )

        suggestion = carbs_run.suggest().suggestion
        assert isinstance(suggestion["torquescale"], float)
        print("Suggestion:", suggestion)
        wandb.config.update(suggestion)

        try:
            losses = test_weld_settings(
                tgt_data_path=pathlib.Path(tgt_data_path),
                env_name=env_name,
                torquescale=suggestion["torquescale"],
            )
        except Exception as e:
            import traceback

            traceback.print_exc()
        else:
            observed_value = losses[TARGET_METRIC]
            obs_out = carbs_run.observe(
                carbs.ObservationInParam(
                    input=suggestion, output=float(observed_value), cost=0
                )
            )

    wandb.agent(sweep_id, _sweep_main, count=TRIALS)


if __name__ == "__main__":
    main()
