# SPDX-License-Identifier: Apache-2.0
import os
import shlex
import subprocess
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def test_launcher_composes_against_pinned_config():
    root = Path(__file__).resolve().parents[2]
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "TRAIN_FILES": "/data/train.parquet",
        "VAL_FILES": "/data/test.parquet",
        "MODEL_PATH": "/models/previous",
    }
    result = subprocess.run(
        ["bash", str(root / "scripts/train_cgl.sh")], env=env, capture_output=True, text=True, check=True
    )
    command = shlex.split(result.stdout)
    assert command[1:3] == ["-m", "verl.trainer.main_ppo"]
    with initialize_config_dir(version_base=None, config_dir=str(root / "verl/trainer/config")):
        config = compose(config_name="ppo_trainer", overrides=command[3:])
    resolved = OmegaConf.to_container(config, resolve=True)
    actor = resolved["actor_rollout_ref"]["actor"]
    assert actor["sft_grpo_loss"] and actor["sft_grpo_mu"]
    assert not actor["sft_grpo_loss_add_linear"]
    assert actor["sft_grpo_routing"] == "paper"
    assert resolved["data"]["train_files"] == "/data/train.parquet"
    assert resolved["custom_reward_function"]["name"] == "compute_score"
    assert not resolved["reward_model"]["launch_reward_fn_async"]
