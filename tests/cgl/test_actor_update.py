# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual actor update body with autograd and a toy CPU actor.

Only backend imports/decorators, batching, and the PPO objective are substituted.
The production update method is compiled unchanged from its AST, allowing us to
test its two backward passes without Ray, CUDA, Ascend, or flash-attention.
This is not an FSDP/distributed execution test.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf


class TensorBatch(dict):
    def split(self, size):
        count = self["responses"].shape[0]
        return [TensorBatch({key: value[i : i + size] for key, value in self.items()}) for i in range(0, count, size)]


class BatchData:
    def __init__(self, tensors):
        self.batch = TensorBatch(tensors)
        self.meta_info = {"temperature": 1.0, "step_now": 1, "entropy_now": 0.1}
        self.non_tensor_batch = {}

    def select(self, batch_keys):
        return BatchData({key: self.batch[key] for key in batch_keys})


def get_update_method(cgl):
    source = Path(__file__).resolve().parents[2] / "verl/workers/actor/dp_actor.py"
    tree = ast.parse(source.read_text())
    actor = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DataParallelPPOActor")
    update = next(node for node in actor.body if isinstance(node, ast.FunctionDef) and node.name == "update_policy")
    update.decorator_list = []

    def fake_ppo_loss(**kwargs):
        loss = -kwargs["log_prob"].mean()
        zero = loss.detach() * 0
        return loss, zero, zero, zero

    def append_metrics(destination, values):
        for key, value in values.items():
            destination.setdefault(key, []).append(value)

    namespace = {
        "torch": torch,
        "DataProto": BatchData,
        "get_device_id": lambda: "cpu",
        "compute_policy_loss": fake_ppo_loss,
        "compute_policy_loss_sft": lambda log_prob_sft, sft_loss_mask, response_mask_new: cgl.supervised_loss(
            log_prob_sft, sft_loss_mask, response_mask_new
        ),
        "entropy_weight": cgl.entropy_weight,
        "combine_gradients": cgl.combine_gradients,
        "append_to_dict": append_metrics,
        "rearrange_micro_batches": lambda batch, max_token_len: (batch.split(2), None),
    }
    module = ast.Module(body=[update], type_ignores=[])
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["update_policy"]


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "enabled,linear,adaptive,selected,expected",
    [
        (True, False, False, True, [0.1, 0.3]),
        (True, True, False, True, [-0.1, 0.3]),
        (False, False, False, True, [0.1, 0.0]),
        (True, False, True, True, [0.1, 0.06]),
        (True, False, False, False, [0.1, 0.0]),
    ],
)
def test_two_backward_passes(cgl, dynamic, enabled, linear, adaptive, selected, expected):
    module = torch.nn.Linear(2, 1, bias=False)
    torch.nn.init.zeros_(module.weight)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    config = OmegaConf.create(
        {
            "sft_grpo_loss": enabled,
            "sft_grpo_loss_add_linear": linear,
            "sft_grpo_mu": adaptive,
            "policy_loss": {"loss_mode": "vanilla"},
            "use_kl_loss": False,
            "ppo_mini_batch_size": 4,
            "ppo_micro_batch_size_per_gpu": 2,
            "ppo_max_token_len_per_gpu": 20,
            "use_dynamic_bsz": dynamic,
            "ppo_epochs": 1,
            "clip_ratio": 0.2,
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.2,
            "entropy_coeff": 0,
            "loss_agg_mode": "token-mean",
        }
    )

    def optimizer_step():
        norm = torch.linalg.vector_norm(module.weight.grad)
        optimizer.step()
        return norm

    actor = SimpleNamespace(
        actor_module=module,
        actor_optimizer=optimizer,
        config=config,
        ulysses_sequence_parallel_size=1,
        _forward_micro_batch=lambda micro_batch, **kwargs: (None, module(micro_batch["input_ids"].float())),
        _optimizer_step=optimizer_step,
    )
    tensors = {
        "responses": torch.ones(4, 1),
        "response_mask": torch.ones(4, 1),
        "input_ids": torch.tensor([[1, 0]] * 4),
        "attention_mask": torch.ones(4, 2),
        "position_ids": torch.zeros(4, 2),
        "old_log_probs": torch.zeros(4, 1),
        "advantages": torch.ones(4, 1),
        "sft_loss_mask": torch.full((4, 1), float(selected)),
        "new_input_ids": torch.tensor([[-2, 3]] * 4),
        "new_attention_mask": torch.ones(4, 2),
        "responses_new": torch.ones(4, 1),
        "new_position_ids": torch.ones(4, 2),
    }
    data = BatchData(tensors)
    if not enabled:
        data.meta_info = {"temperature": 1.0}
    metrics = get_update_method(cgl)(actor, data)
    torch.testing.assert_close(module.weight.detach().flatten(), torch.tensor(expected))
    assert metrics["actor/grad_norm"][0] > 0
