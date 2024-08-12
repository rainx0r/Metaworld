import pathlib
import sys

import numpy as np
import numpy.typing as npt
import polars as pl


def get_mse_loss(
    src: npt.NDArray[np.float64], tgt: npt.NDArray[np.float64]
) -> np.float64:
    return ((src - tgt) ** 2).mean()


def get_mae_loss(
    src: npt.NDArray[np.float64], tgt: npt.NDArray[np.float64]
) -> np.float64:
    return np.absolute(src - tgt).mean()


def get_loss_for_key(
    src: pl.DataFrame, tgt: pl.DataFrame, key: str
) -> tuple[np.float64, np.float64]:
    src_data = src[key].to_numpy()
    tgt_data = tgt[key].to_numpy()
    return get_mse_loss(src_data, tgt_data), get_mae_loss(src_data, tgt_data)


def main() -> None:
    if not len(sys.argv) > 2:
        raise ValueError(
            "You need to provide a source file path and a target file path for comparison."
        )

    src_file = pathlib.Path(sys.argv[1])
    tgt_file = pathlib.Path(sys.argv[2])

    src_data = pl.read_parquet(src_file)
    tgt_data = pl.read_parquet(tgt_file)

    src_actions = src_data["action"].to_numpy()
    tgt_actions = tgt_data["action"].to_numpy()
    assert np.allclose(src_actions, tgt_actions)

    obs_mse_loss, obs_mae_loss = get_loss_for_key(src_data, tgt_data, "obs")
    rew_mse_loss, rew_mae_loss = get_loss_for_key(src_data, tgt_data, "reward")

    print("Observation loss (MSE):", obs_mse_loss)
    print("Observation loss (MAE):", obs_mae_loss)
    print("Reward loss (MSE):", rew_mse_loss)
    print("Reward loss (MAE):", rew_mae_loss)


if __name__ == "__main__":
    main()
