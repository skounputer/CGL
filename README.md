<div align="center">

# CGL: Advancing Continual GUI Learning via Reinforcement Fine-Tuning

### CVPR 2026

<a href="https://openaccess.thecvf.com/content/CVPR2026/html/Yao_CGL_Advancing_Continual_GUI_Learning_via_Reinforcement_Fine-Tuning_CVPR_2026_paper.html"><img src="https://img.shields.io/badge/Paper-CVPR%202026-b31b1b" alt="CVPR 2026 paper"></a>
<a href="https://huggingface.co/datasets/Regimes/androidcontrol-cl"><img src="https://img.shields.io/badge/Dataset-%F0%9F%A4%97%20AndroidControl--CL-yellow" alt="AndroidControl-CL on Hugging Face"></a>
<a href="https://github.com/skounputer/CGL"><img src="https://img.shields.io/badge/Code-GitHub-black?logo=github" alt="Code"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue" alt="Apache 2.0 license"></a>

[English](README.md) | [中文](README.zh-CN.md)

</div>

## Overview

**CGL** studies continual learning for GUI agents: adapting to new application domains while retaining previously acquired interaction skills. It brings together the adaptation efficiency of supervised fine-tuning and the retention behavior of reinforcement learning through three components:

1. **Error-Aware Routing:** introduce corrected-response supervision when a rollout group fails to reach the full action reward.
2. **Entropy-Regulated Tuning:** adjust the supervised objective's weight using policy entropy.
3. **Conditional Gradient Surgery:** project conflicting supervised gradients relative to the GRPO update.

We also introduce **AndroidControl-CL**, a benchmark that organizes Android GUI interactions into seven application categories for sequential training and evaluation.

## News

- **[2026.10.16]** CGL training code and launch scripts are available. Access AndroidControl-CL on [Hugging Face](https://huggingface.co/datasets/Regimes/androidcontrol-cl).
- **[CVPR 2026]** Read the [conference paper](https://openaccess.thecvf.com/content/CVPR2026/html/Yao_CGL_Advancing_Continual_GUI_Learning_via_Reinforcement_Fine-Tuning_CVPR_2026_paper.html).

## Dataset

The [AndroidControl-CL dataset](https://huggingface.co/datasets/Regimes/androidcontrol-cl) includes screenshots, task annotations, and category-specific training/test files.

| Task category | Parquet filename prefix |
| --- | --- |
| Shopping | `shopping` |
| Productivity and Office | `productivity` |
| Communication | `communication` |
| Travel and Transportation | `travel_and_local` |
| System Tools | `tools` |
| Education and Science | `education` |
| Life and Entertainment | `entertainment` |

Download the files for the current task from `parquet_convert/`, for example `productivity_train.parquet` and `productivity_test.parquet`. See [training notes](docs/TRAINING.md#data-and-training) for the input schema and supported actions.

## Installation

The original experiments use **8 Ascend NPUs and vLLM 0.7.0**. Prepare the matching CANN, PyTorch/torch-npu, and vllm-ascend environment, then install:

```bash
git clone https://github.com/skounputer/CGL.git
cd CGL
pip install -r requirements-npu.txt
pip install tensorboard
pip install -e . --no-deps
```

This repository includes the verl `0.4.1.dev` backbone. The Ascend installation uses `requirements-npu.txt`; the upstream `[vllm]` extra is intended for CUDA installations. Additional backend details are in the [training notes](docs/TRAINING.md#environment).

## Training

Set the current task's data and the initial or previous-task Hugging Face checkpoint:

```bash
export TRAIN_FILES=/path/to/productivity_train.parquet
export VAL_FILES=/path/to/productivity_test.parquet
export MODEL_PATH=/path/to/previous_task_hf_checkpoint

bash scripts/train_cgl.sh
```

Use `DRY_RUN=1 bash scripts/train_cgl.sh` to inspect the command. For continual learning, train one task, [export its actor checkpoint](docs/TRAINING.md#data-and-training), and pass the resulting Hugging Face model as `MODEL_PATH` for the next task.

### Ablations

| Setting | Override passed to the training script |
| --- | --- |
| CGL | Default launcher configuration |
| Direct gradient addition | `actor_rollout_ref.actor.sft_grpo_loss_add_linear=True` |
| Fixed supervised weight | `actor_rollout_ref.actor.sft_grpo_mu=False` |
| GRPO only | `actor_rollout_ref.actor.sft_grpo_loss=False` |

```bash
bash scripts/train_cgl.sh actor_rollout_ref.actor.sft_grpo_mu=False
```

## Reproducibility

The CPU checks cover CGL's routing, loss, gradient update, reward callback, and configuration:

```bash
pip install -r requirements-cgl-test.txt
pytest -q tests/cgl
```

Full NPU training revalidation is pending for this release. Initial SFT, model weights, and the complete evaluation pipeline are not included. See [validation details](docs/VALIDATION.md) and [implementation changes](docs/MIGRATION.md).

## Citation

If you use CGL or AndroidControl-CL, please cite:

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

## Acknowledgements and License

Our training infrastructure builds on [verl / HybridFlow](https://github.com/verl-project/verl). The code is released under [Apache-2.0](LICENSE), with upstream [notices](Notice.txt) retained. Dataset and model licenses apply separately.
