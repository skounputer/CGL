# Source reconstruction and release changes

## Matching upstream baseline

Source: <https://github.com/verl-project/verl>

Pinned commit: `3f929af747777f43041b8c6c56da26c909842e27` (July 7, 2025).

Version file at that commit: `0.4.1.dev`.

The supplied folder contained four Python files saved with `.tex` extensions, one actor YAML file, one shell launcher saved as `.tex`, and an unrelated LaTeX skeleton. There was no Git history, package metadata, environment lock, or complete repository. The original files remain outside this repository in the local source folder. Their hashes are recorded in [SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json).

The selected baseline's `actor.yaml` differs from the supplied configuration only by the three added CGL flags. Its core algorithms match except for an unused debug import and the two added supervised-loss functions. The actor's forward path matches, with CGL changes concentrated in `update_policy`. These matches establish a compatible reconstruction baseline, but do not identify the lost checkout's exact HEAD.

The repository's first commit imports a self-contained upstream source snapshot, with its original license and notices. Later commits integrate CGL. This repository does not retain upstream's complete Git history. The original upstream README is `README.verl.md`; upstream examples and tests remain available. The original checkpoint converter is retained as `scripts/legacy_model_merger.py`.

## File mapping

| Supplied file | Integrated location |
| --- | --- |
| `core_algos.tex` | `verl/trainer/ppo/core_algos.py`, `verl/trainer/ppo/cgl.py` |
| `dpactor.tex` | `verl/workers/actor/dp_actor.py` |
| `ray_trainer.tex` | `verl/trainer/ppo/ray_trainer.py`, `verl/trainer/ppo/cgl.py` |
| `actor.yaml` | `verl/trainer/config/actor/actor.yaml` |
| `andriod_control_reward.tex` | `verl/utils/reward_score/android_control.py` |
| `nlinear.tex` | `scripts/train_cgl.sh` |

## Algorithm behavior retained

- Correct one well-formed `<action>...</action>` in a generated response while retaining the surrounding text. Missing or multiple action tags do not receive supervision.
- Spatial demonstrations use the supplied point or Gaussian rejection sampling within the target box.
- The supervised loss averages selected non-padding tokens, with denominator epsilon `1e-8`.
- Entropy weight is `step/5` when `step <= 5` and entropy is at most `0.4`; otherwise `exp(min(20*entropy - 10, 0))`.
- GRPO and supervised gradients use separate backward passes. Conflict projection is applied per local parameter/shard when their dot product is negative. This retains the supplied implementation's granularity; it is not a full-model, globally reduced gradient projection.

## Explicit behavioral corrections

1. **Reward aggregation and group identity.** The old trainer indexed `reward_tensor[j, 0]`, although the naive reward manager stores rewards on the last valid response token. It also assumed that each group occupied adjacent rows after batch balancing. The integrated version sums rewards over tokens and groups by UID.
2. **Routing threshold.** The paper's section 5.1 triggers SFT when no rollout achieves the full action reward. The default `sft_grpo_routing=paper` uses maximum group reward `< 2`. `legacy_nonzero` uses the supplied nonzero threshold with repaired indexing/grouping. It is not a bug-for-bug replay mode.
3. **Corrected sequence construction.** Use actual prompt length, right-pad teacher-forced responses, extend position IDs from the prompt (including multimodal position channels), and restore tokenizer padding afterward. The supplied code reused rollout positions and a configured prompt-length slice. Demonstrations that would be truncated are excluded from the supervised loss.
4. **Disabled-SFT regression.** Do not clear policy gradients merely because the projection flag is false when supervision itself is disabled. Missing CGL metadata is accepted in GRPO-only runs.
5. **Dynamic microbatch scaling.** Use the response tensor's sample count rather than `len(dict)`, which counts fields.
6. **Reward integration.** Add the `data_source, solution_str, ground_truth, extra_info` callback expected by verl. The supplied launcher did not configure a custom reward callback.
7. **Parsing.** Replace coordinate `eval` with literal parsing; reject malformed/non-finite coordinates; require complete action-string matches. Retain the original scoring thresholds and open-box boundary rule. Invalid or degenerate corrective boxes fail clearly rather than looping forever.
8. **Projection arithmetic.** Accumulate dot products and squared norms in float32, then cast back to the gradient dtype, avoiding low-precision norm underflow. The projection remains local and parameter-wise.
9. **Unsupported modes.** Reject CGL with asynchronous rewards, Megatron actors, or non-vanilla policy losses rather than reaching undefined state. Correct the inherited non-vanilla policy branch's `response_mask=advantages` argument to use the response mask.

These fixes can affect optimization trajectories and must not be presented as verification of historical experimental results.

## Publication cleanup

Machine-specific working directories, dataset/checkpoint paths, unused diagnostics, debug tensor dumps, commented experiments, and unrelated document scaffolding are not included in the CGL integration. The launcher uses explicit environment variables and an AndroidControl validation file instead of the unrelated Geo3K validation path in the supplied script. The CGL configuration defaults to off for inherited upstream recipes; the new launcher turns it on.

Apache-2.0, upstream notices, and existing source copyright headers are retained. New algorithm helpers carry a CGL contributor notice. Dataset and model licensing remains separate. See [the paper](https://openaccess.thecvf.com/content/CVPR2026/html/Yao_CGL_Advancing_Continual_GUI_Learning_via_Reinforcement_Fine-Tuning_CVPR_2026_paper.html) and [the public dataset](https://huggingface.co/datasets/Regimes/androidcontrol-cl) for those artifacts.

## Validation boundary

CPU tests execute the algorithm kernels, reward adapter, corrected-batch assembly, and the production actor update body with a toy autograd actor. The actor harness substitutes backend imports, batching, and a simple differentiable policy objective; it does not run FSDP. Tests also compose the launcher's exact Hydra overrides against the pinned configuration. Syntax and lint checks cover the modified code.

Full training, distributed gradient behavior, GPU/NPU kernels, real tokenizer/model/image compatibility, and checkpoint conversion require validation on the original class of hardware. Exact historical package versions, initial SFT preparation, authoritative task-order manifests, and complete continual-learning evaluation scripts were not supplied. This release makes those omissions explicit instead of inventing reproduction details.
