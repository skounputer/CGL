# CGL

Code for **CGL: Advancing Continual GUI Learning via Reinforcement Fine-Tuning**.

[Paper](https://openaccess.thecvf.com/content/CVPR2026/html/Yao_CGL_Advancing_Continual_GUI_Learning_via_Reinforcement_Fine-Tuning_CVPR_2026_paper.html) · [AndroidControl-CL dataset](https://huggingface.co/datasets/Regimes/androidcontrol-cl) · [中文说明](README.zh-CN.md)

This initial code release integrates the supplied CGL research files into a matching verl backbone. CPU algorithm tests and configuration checks are provided. **End-to-end GPU/Ascend training and the paper's numerical results have not yet been revalidated for this assembled release.** Model weights, the initial SFT pipeline, and the complete evaluation pipeline are not included.

CGL combines a GRPO policy update with corrected-response supervision, an entropy-dependent supervision weight, and conditional gradient projection. The implementation retains the original three switches:

| Actor option | Meaning | CGL launcher |
| --- | --- | --- |
| `sft_grpo_loss` | Enable corrected-response supervision | `True` |
| `sft_grpo_mu` | Apply entropy-dependent SFT weighting | `True` |
| `sft_grpo_loss_add_linear` | Add gradients directly; `False` enables conflict projection | `False` |

## Version and scope

The backbone is pinned to [verl commit `3f929af747777f43041b8c6c56da26c909842e27`](https://github.com/verl-project/verl/commit/3f929af747777f43041b8c6c56da26c909842e27), dated July 7, 2025; its version file reads **`0.4.1.dev`**. This is a matching reconstruction baseline, not proof of the missing original checkout's exact HEAD. It is not a port to current verl. Upstream package sources, examples, and tests are retained; see [the original README](README.verl.md).

CGL is wired into the **FSDP actor with vanilla policy loss and synchronous rewards**. Megatron and asynchronous reward execution are rejected when CGL is enabled. Projection follows the supplied implementation's **local, parameter/shard-wise** rule; it is not a distributed projection over one global model gradient vector. FSDP2 and other inherited upstream backends still need hardware validation.

## Environment

The original experiments used **eight Ascend NPUs with vLLM 0.7.0**. Use a compatible CANN, PyTorch/torch-npu, and vllm-ascend environment. Python 3.10 is the recommended starting point.

For the original Ascend path, first install the matching PyTorch/torch-npu, CANN, and vllm-ascend stack. Then, inside that environment:

```bash
pip install -r requirements-npu.txt
pip install tensorboard
pip install -e . --no-deps
```

Use `requirements-npu.txt` for the Ascend installation path. The inherited `[vllm]` package extra and CUDA dependency recipes target upstream CUDA installations. CUDA users should start with the [upstream installation instructions](README.verl.md) for this historical version, then use `DEVICE=cuda` with the CGL launcher.

Install **this checkout** as the `verl` package; do not separately install a current PyPI verl into the same environment.

## Data and training

Obtain the dataset separately from [Regimes/androidcontrol-cl](https://huggingface.co/datasets/Regimes/androidcontrol-cl). For example, download its `parquet_convert/productivity_train.parquet` and `productivity_test.parquet` files. Weights and datasets are excluded from Git.

The trainer expects verl-format Parquet rows containing:

- `prompt`: chat messages, with the image placeholder expected by the model's processor.
- `images`: multimodal images supported by the pinned verl data loader.
- `data_source`: reward-source label.
- `reward_model.ground_truth`: the target action parameter.
- `extra_info.action_type` and `extra_info.answer`: action type and corrective target.

Spatial answers use normalized point `[x,y]` or box `[left,top,right,bottom]` coordinates. Supported actions are `Click`, `Long_press`, `Scroll`, `Open`, `Input_text`, `Back`, `Home`, `Wait`, and `Finish`. Predictions use `<action>...</action>` tags. The released data schema and image loading should be checked in your training environment before launching a full run.

```bash
export TRAIN_FILES=/path/to/productivity_train.parquet
export VAL_FILES=/path/to/productivity_test.parquet
export MODEL_PATH=/path/to/initial_sft_or_previous_task_hf_checkpoint

# Inspect the command without starting Ray or allocating devices.
DRY_RUN=1 bash scripts/train_cgl.sh

# Source your Ascend vendor environment before launching.
bash scripts/train_cgl.sh
```

Paths above must be replaced with local files. Optional environment variables include `DEVICE`, `ENGINE`, `N_GPUS`, `TP_SIZE`, `OUTPUT_DIR`, and `EXPERIMENT_NAME`. Additional Hydra options may be passed at the end, for example:

```bash
# Smoke run: adjust batch sizes/device count for available hardware as needed.
bash scripts/train_cgl.sh trainer.total_training_steps=1 trainer.save_freq=1

# Direct gradient addition ablation.
bash scripts/train_cgl.sh actor_rollout_ref.actor.sft_grpo_loss_add_linear=True

# GRPO-only ablation.
bash scripts/train_cgl.sh actor_rollout_ref.actor.sft_grpo_loss=False
```

For continual training, finish one task, export its actor checkpoint to Hugging Face format, and pass that model as `MODEL_PATH` for the next task. The pinned upstream converter is included (not hardware-tested in this release):

```bash
python scripts/legacy_model_merger.py merge --backend fsdp \
  --local_dir outputs/cgl_androidcontrol/global_step_10/actor \
  --target_dir models/task2_hf
```

Replace the checkpoint step/path with your completed task's checkpoint. The launcher trains one task per invocation. The supplied files did not contain authoritative task-order manifests or the complete evaluation program; those are still needed to reproduce every paper table.

## Corrections relative to the supplied files

The default `sft_grpo_routing=paper` selects a prompt group when **none of its rollouts reaches reward 2**. The supplied file instead used nonzero reward checks at tensor column zero. The release sums token rewards and groups by prompt UID, including after batch balancing. `CGL_ROUTING=legacy_nonzero` retains the old nonzero threshold with corrected indexing; it does not reproduce the original indexing bug.

Other changes include a verl-compatible reward callback, safe coordinate parsing, corrected-response position IDs/right padding, exclusion of truncated demonstrations, correct dynamic microbatch sample counts, and retaining GRPO gradients when the SFT branch is disabled. See [migration details](docs/MIGRATION.md) for provenance and behavioral differences. These corrections may change training trajectories relative to an old experimental run.

## CPU checks

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-cgl-test.txt
pytest -q tests/cgl
python -m compileall -q verl
bash -n scripts/train_cgl.sh
```

These checks cover algorithm kernels, reward parsing, and corrected-batch assembly. They do not validate device kernels, distributed execution, model/data compatibility, or reproduce training scores. Inherited upstream tests under other directories may require additional dependencies, models, and hardware.

## Citation and license

```bibtex
@InProceedings{Yao_2026_CVPR,
  author={Yao, Zhenquan and Huang, Zitong and Zeng, Yihan and Han, Jianhua and Xu, Hang and Feng, Chun-Mei and Ma, Jianwei and Zuo, Wangmeng},
  title={CGL: Advancing Continual GUI Learning via Reinforcement Fine-Tuning},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month={June},
  year={2026},
  pages={15212-15221},
  url={https://openaccess.thecvf.com/content/CVPR2026/html/Yao_CGL_Advancing_Continual_GUI_Learning_via_Reinforcement_Fine-Tuning_CVPR_2026_paper.html}
}
```

The code is distributed under [Apache-2.0](LICENSE), retaining upstream copyright and [notices](Notice.txt). Please also cite [verl / HybridFlow](https://github.com/verl-project/verl) when using its training infrastructure. Dataset and model licenses are separate from the code license.
