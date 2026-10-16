# SPDX-License-Identifier: Apache-2.0
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.mark.parametrize("step,entropy,expected", [(1, 0.1, 0.2), (5, 0.4, 1), (6, 0.5, 1), (9, 0, math.exp(-10))])
def test_entropy_schedule(cgl, step, entropy, expected):
    assert cgl.entropy_weight(step, entropy) == pytest.approx(expected)


@pytest.mark.parametrize(
    "sft,policy,expected",
    [
        ([2, 3], [1, 0], [3, 3]),
        ([-2, 3], [1, 0], [1, 3]),
        ([0, 2], [1, 0], [1, 2]),
        ([2, 3], [0, 0], [2, 3]),
    ],
)
def test_gradient_projection(cgl, sft, policy, expected):
    sft, policy = torch.tensor(sft, dtype=torch.float32), torch.tensor(policy, dtype=torch.float32)
    combined = cgl.combine_gradients(sft, policy)
    torch.testing.assert_close(combined, torch.tensor(expected, dtype=torch.float32))
    assert torch.dot(combined - policy, policy) >= -1e-6


def test_missing_gradient_and_low_precision(cgl):
    grad = torch.tensor([1.0, 0.0])
    assert cgl.combine_gradients(None, grad) is grad
    assert cgl.combine_gradients(grad, None) is grad
    assert cgl.combine_gradients(None, None) is None
    policy = torch.tensor([1e-4, 0], dtype=torch.bfloat16)
    sft = torch.tensor([-1e-3, 2e-4], dtype=torch.bfloat16)
    out = cgl.combine_gradients(sft, policy)
    assert out.dtype == policy.dtype and torch.isfinite(out).all()
    assert out[0] > 0


def test_masked_sft_loss_and_backward(cgl):
    logp = torch.tensor([[-1.0, -3, -99], [-20, -20, -20]], requires_grad=True)
    loss = cgl.supervised_loss(logp, torch.tensor([[1], [0]]), torch.tensor([[1, 1, 0], [1, 1, 1]]))
    assert loss.item() == pytest.approx(2)
    loss.backward()
    torch.testing.assert_close(logp.grad, torch.tensor([[-0.5, -0.5, 0], [0.0, 0, 0]]))


def test_no_sft_samples_has_zero_gradient(cgl):
    logp = torch.randn(2, 5, requires_grad=True)
    loss = cgl.supervised_loss(logp, torch.zeros(2, 1), torch.ones(2, 5))
    loss.backward()
    assert loss.item() == 0 and torch.count_nonzero(logp.grad) == 0


def test_empty_half_precision_supervision_stays_finite(cgl):
    logp = torch.zeros(2, 5, dtype=torch.float16, requires_grad=True)
    loss = cgl.supervised_loss(logp, torch.zeros(2, 1), torch.ones(2, 5))
    assert torch.isfinite(loss) and loss.item() == 0
    loss.backward()
    assert torch.isfinite(logp.grad).all()


def test_routing_uses_uid_after_reordering(cgl):
    responses = ["<action>Back()</action>"] * 4
    infos = [{"action_type": "Home"}] * 4
    _, selected = cgl.rewrite_responses(responses, [0, 0, 2, 1], ["A", "B", "A", "B"], infos)
    assert selected == [False, True, False, True]


def test_paper_and_legacy_thresholds(cgl):
    args = (["<action>Back()</action>"], [1], ["A"], [{"action_type": "Home"}])
    assert cgl.rewrite_responses(*args)[1] == [True]
    assert cgl.rewrite_responses(*args, routing="legacy_nonzero")[1] == [False]


def test_correction_preserves_reasoning_and_literal_backslashes(cgl):
    response = "<think>reason</think><action>Input_text(wrong)</action>"
    info = {"action_type": "Input_text", "answer": r"C:\new\1"}
    corrected, selected = cgl.rewrite_responses([response], [0], ["x"], [info])
    assert corrected == [r"<think>reason</think><action>Input_text(C:\new\1)</action>"]
    assert selected == [True]


def test_missing_or_multiple_actions_are_not_selected(cgl):
    out, selected = cgl.rewrite_responses(
        ["no action", "<action>Back()</action><action>Home()</action>"],
        [0, 0],
        ["x", "x"],
        [{"action_type": "Home"}] * 2,
    )
    assert selected == [False, False] and out[0] == "no action"


def test_spatial_sampling(cgl):
    rng = np.random.default_rng(7)
    for _ in range(30):
        x, y = cgl.sample_target_point("[0.1, 0.2, 0.3, 0.4]", rng)
        assert 0.1 < x < 0.3 and 0.2 < y < 0.4
    assert cgl.sample_target_point("[0.2, 0.3]") == (0.2, 0.3)


@pytest.mark.parametrize("answer", ["[0, 0, 0, 1]", "[1, 0, 0, 1]", "[1]", "__import__('os').system('false')"])
def test_invalid_spatial_answers_rejected(cgl, answer):
    with pytest.raises(ValueError):
        cgl.sample_target_point(answer)


@pytest.mark.parametrize("multimodal", [False, True])
def test_corrected_positions(cgl, multimodal):
    positions = torch.tensor([[0, 0, 1, 2, 90, 90]])
    if multimodal:
        positions = positions.unsqueeze(1).repeat(1, 3, 1)
    actual = cgl.corrected_position_ids(positions, 4, 2)
    torch.testing.assert_close(actual[..., -2:], torch.full_like(positions[..., :2], 3) + torch.tensor([0, 1]))
    torch.testing.assert_close(actual[..., :4], positions[..., :4])


class CharacterTokenizer:
    """A deterministic tokenizer for tensor assembly tests; no model download."""

    padding_side = "left"

    def encode(self, text, **kwargs):
        return list(text.encode("ascii"))

    def decode(self, tokens, **kwargs):
        return bytes(tokens).decode("ascii")

    def __call__(self, texts, max_length, **kwargs):
        assert self.padding_side == "right"
        rows = [self.encode(text)[:max_length] for text in texts]
        return {
            "input_ids": torch.tensor([row + [0] * (max_length - len(row)) for row in rows]),
            "attention_mask": torch.tensor([[1] * len(row) + [0] * (max_length - len(row)) for row in rows]),
        }


def make_batch(response_length=64, answer="Home"):
    text = "<action>Back()</action>"
    tokens = list(text.encode("ascii"))
    batch = {
        "responses": torch.tensor([tokens + [0] * (response_length - len(tokens))]),
        "response_mask": torch.tensor([[1] * len(tokens) + [0] * (response_length - len(tokens))]),
        "prompts": torch.tensor([[0, 4, 5]]),
        "attention_mask": torch.tensor([[0, 1, 1] + [1] * len(tokens) + [0] * (response_length - len(tokens))]),
        "position_ids": torch.arange(response_length + 3).unsqueeze(0),
    }
    info = {"action_type": answer, "answer": "a" * 100}
    return SimpleNamespace(batch=batch, non_tensor_batch={"uid": ["A"], "extra_info": [info]})


def test_batch_assembly_and_terminal_token_reward(cgl):
    data = make_batch()
    tokenizer = CharacterTokenizer()
    reward = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
    reward[0, 20] = 2
    cgl.prepare_supervised_batch(data, reward, tokenizer, {})
    assert data.batch["sft_loss_mask"].item() == 0  # full score is not at column zero
    assert tokenizer.padding_side == "left"
    assert data.batch["new_input_ids"].shape == data.batch["new_position_ids"].shape
    data = make_batch()
    cgl.prepare_supervised_batch(data, reward * 0, tokenizer, {})
    assert data.batch["sft_loss_mask"].item() == 1
    length = int(data.batch["new_attention_mask"][0, 3:].sum())
    assert tokenizer.decode(data.batch["responses_new"][0, :length].tolist()) == "<action>Home()</action>"


def test_truncated_demonstration_not_supervised(cgl):
    data = make_batch(answer="Input_text")
    reward = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
    cgl.prepare_supervised_batch(data, reward, CharacterTokenizer(), {})
    assert data.batch["sft_loss_mask"].item() == 0
