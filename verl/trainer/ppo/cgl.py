# Copyright 2026 CGL contributors
# SPDX-License-Identifier: Apache-2.0
"""CGL routing, entropy weighting, and parameter-wise gradient surgery.

The constants and projection granularity follow the supplied research code.
Distributed, global-vector projection is not implemented here.
"""

import ast
import math
import re
from collections import defaultdict

import numpy as np
import torch


def entropy_weight(step, entropy):
    """Original five-step warmup followed by entropy-dependent exponential decay."""
    if step <= 5 and entropy <= 0.40:
        return step / 5.0
    return math.exp(min(20.0 * entropy - 10.0, 0.0))


def combine_gradients(sft_grad, policy_grad):
    """Project conflicting SFT components off a policy gradient, then add them.

    Called once per local parameter/shard, matching the research implementation.
    Reductions use float32 to avoid low-precision norm underflow.
    """
    if sft_grad is None:
        return policy_grad
    if policy_grad is None:
        return sft_grad
    sft = sft_grad.float()
    policy = policy_grad.float()
    dot = torch.sum(sft * policy)
    norm_sq = torch.sum(policy.square())
    if dot < 0 and norm_sq > 0:
        sft = sft - (dot / norm_sq) * policy
    return (sft + policy).to(policy_grad.dtype)


def supervised_loss(log_probs, sample_mask, response_mask):
    """Token mean over selected demonstrations; zero when none are selected."""
    mask = sample_mask.reshape(-1, 1).float() * response_mask.float()
    return -(log_probs * mask).sum() / (mask.sum() + 1e-8)


def sample_target_point(answer, rng=None):
    """Sample a point inside a target box, or return an annotated target point."""
    target = ast.literal_eval(answer) if isinstance(answer, str) else answer
    if not isinstance(target, (list, tuple)) or len(target) not in (2, 4):
        raise ValueError("Spatial answers must contain two coordinates or a four-coordinate box")
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in target):
        raise ValueError("Spatial answers must contain finite numeric coordinates")
    if len(target) == 2:
        return tuple(target)
    left, top, right, bottom = target
    if left >= right or top >= bottom:
        raise ValueError("Target bounding boxes must have positive width and height")
    rng = np.random if rng is None else rng
    center = [(left + right) / 2, (top + bottom) / 2]
    half = [(right - left) / 2, (bottom - top) / 2]
    for _ in range(10000):
        point = rng.multivariate_normal(center, np.diag(np.square(half)))
        if left < point[0] < right and top < point[1] < bottom:
            rounded = tuple(round(float(v), 4) for v in point)
            if left < rounded[0] < right and top < rounded[1] < bottom:
                return rounded
            return tuple(float(v) for v in point)
    raise RuntimeError("Unable to sample a point inside the target bounding box")


def rewrite_responses(responses, scores, group_ids, extra_infos, routing="paper", success_reward=2.0, rng=None):
    """Correct single-action responses in unsuccessful prompt groups.

    ``paper`` triggers when no rollout reaches the full reward. ``legacy_nonzero``
    reproduces the supplied nonzero-score threshold, with corrected score/group
    indexing. It does not reproduce the original first-token indexing bug.
    """
    if routing not in ("paper", "legacy_nonzero"):
        raise ValueError(f"Unsupported CGL routing: {routing}")
    if not (len(responses) == len(scores) == len(group_ids) == len(extra_infos)):
        raise ValueError("Responses, scores, group IDs, and metadata must have equal lengths")
    group_scores = defaultdict(list)
    for uid, score in zip(group_ids, scores):
        group_scores[uid].append(float(score))
    pattern = re.compile(r"<action>.*?</action>", re.DOTALL)
    corrected = list(responses)
    selected = [False] * len(responses)
    for i, (response, uid, info) in enumerate(zip(responses, group_ids, extra_infos)):
        successful = (
            max(group_scores[uid]) >= success_reward
            if routing == "paper"
            else any(score != 0 for score in group_scores[uid])
        )
        if successful or len(pattern.findall(response)) != 1:
            continue
        action_type = info["action_type"]
        if action_type in ("Click", "Long_press"):
            x, y = sample_target_point(info["answer"], rng=rng)
            action = f"{action_type}({x},{y})"
        elif action_type in ("Open", "Input_text", "Scroll"):
            action = f"{action_type}({info['answer']})"
        elif action_type in ("Back", "Home", "Wait", "Finish"):
            action = f"{action_type}()"
        else:
            raise ValueError(f"Unsupported ground-truth action type: {action_type}")
        corrected[i] = pattern.sub(lambda _, action=action: f"<action>{action}</action>", response)
        selected[i] = True
    return corrected, selected


def corrected_position_ids(position_ids, prompt_length, response_length):
    """Extend text or multimodal prompt positions for teacher-forced responses."""
    prompt = position_ids[..., :prompt_length]
    offsets = torch.arange(1, response_length + 1, device=position_ids.device, dtype=position_ids.dtype)
    response = prompt[..., -1:] + offsets
    return torch.cat((prompt, response), dim=-1).contiguous()


def prepare_supervised_batch(data, token_rewards, tokenizer, config):
    """Attach corrected examples without assuming contiguous rollout groups."""
    batch = data.batch
    response_length = batch["responses"].shape[-1]
    lengths = batch["response_mask"].sum(-1).tolist()
    responses = [
        tokenizer.decode(row[: int(length)], skip_special_tokens=False)
        for row, length in zip(batch["responses"].cpu().tolist(), lengths)
    ]
    corrected, selected = rewrite_responses(
        responses,
        token_rewards.sum(-1).detach().cpu().tolist(),
        data.non_tensor_batch["uid"],
        data.non_tensor_batch["extra_info"],
        routing=config.get("sft_grpo_routing", "paper"),
        success_reward=config.get("sft_grpo_success_reward", 2.0),
    )
    # Do not train on a target whose action/EOS would be cut by truncation.
    for i, text in enumerate(corrected):
        if selected[i] and len(tokenizer.encode(text, add_special_tokens=False)) > response_length:
            selected[i] = False
    previous_padding = tokenizer.padding_side
    try:
        tokenizer.padding_side = "right"
        encoded = tokenizer(
            corrected,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=response_length,
            add_special_tokens=False,
        )
    finally:
        tokenizer.padding_side = previous_padding
    device = batch["prompts"].device
    responses_new = encoded["input_ids"].to(device)
    prompt_length = batch["prompts"].shape[-1]
    batch["responses_new"] = responses_new
    batch["new_input_ids"] = torch.cat((batch["prompts"], responses_new), dim=-1).contiguous()
    batch["new_attention_mask"] = torch.cat(
        (batch["attention_mask"][:, :prompt_length], encoded["attention_mask"].to(device)), dim=-1
    ).contiguous()
    batch["new_position_ids"] = corrected_position_ids(batch["position_ids"], prompt_length, response_length)
    batch["sft_loss_mask"] = torch.tensor(selected, dtype=torch.float32, device=device).unsqueeze(-1)
