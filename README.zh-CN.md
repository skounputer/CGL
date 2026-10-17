<div align="center">

# CGL: Advancing Continual GUI Learning via Reinforcement Fine-Tuning

### CVPR 2026

<a href="https://openaccess.thecvf.com/content/CVPR2026/html/Yao_CGL_Advancing_Continual_GUI_Learning_via_Reinforcement_Fine-Tuning_CVPR_2026_paper.html"><img src="https://img.shields.io/badge/Paper-CVPR%202026-b31b1b" alt="CVPR 2026 论文"></a>
<a href="https://huggingface.co/datasets/Regimes/androidcontrol-cl"><img src="https://img.shields.io/badge/Dataset-%F0%9F%A4%97%20AndroidControl--CL-yellow" alt="AndroidControl-CL 数据集"></a>
<a href="https://github.com/skounputer/CGL"><img src="https://img.shields.io/badge/Code-GitHub-black?logo=github" alt="代码"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue" alt="Apache 2.0"></a>

[English](README.md) | [中文](README.zh-CN.md)

</div>

## 项目介绍

**CGL** 面向 GUI 智能体的持续学习：在学习新应用领域的同时，保留已掌握的交互技能。方法结合监督微调的适应效率和强化学习的知识保持能力，包含三个模块：

1. **错误感知路由**：同一目标的一组采样均未达到满分时，引入纠正回答的监督学习。
2. **熵调控训练**：根据策略熵调整监督损失权重。
3. **条件梯度手术**：以 GRPO 更新为参照，对冲突的监督梯度进行投影。

我们同时提出 **AndroidControl-CL**，将 Android GUI 交互划分为七类应用任务，用于顺序训练和持续学习评测。

## 项目动态

- **[2026.10.16]** 训练代码与启动脚本已发布。AndroidControl-CL 数据集可在 [Hugging Face](https://huggingface.co/datasets/Regimes/androidcontrol-cl) 下载。
- **[CVPR 2026]** [会议论文](https://openaccess.thecvf.com/content/CVPR2026/html/Yao_CGL_Advancing_Continual_GUI_Learning_via_Reinforcement_Fine-Tuning_CVPR_2026_paper.html)已公开。

## 数据集

[AndroidControl-CL](https://huggingface.co/datasets/Regimes/androidcontrol-cl) 包含截图、任务标注，以及按应用类别划分的训练／测试文件。

| 任务类别 | Parquet 文件名前缀 |
| --- | --- |
| 购物 | `shopping` |
| 生产力与办公 | `productivity` |
| 通信 | `communication` |
| 出行与交通 | `travel_and_local` |
| 系统工具 | `tools` |
| 教育与科学 | `education` |
| 生活与娱乐 | `entertainment` |

从数据仓库的 `parquet_convert/` 下载当前任务对应文件，例如 `productivity_train.parquet` 和 `productivity_test.parquet`。输入字段与动作格式见[训练说明](docs/TRAINING.md#data-and-training)。

## 环境安装

原始实验使用 **8 张 Ascend NPU 和 vLLM 0.7.0**。准备配套的 CANN、PyTorch/torch-npu、vllm-ascend 环境后安装：

```bash
git clone https://github.com/skounputer/CGL.git
cd CGL
pip install -r requirements-npu.txt
pip install tensorboard
pip install -e . --no-deps
```

本仓库包含 verl `0.4.1.dev` 底座。Ascend 环境使用 `requirements-npu.txt`，继承的 `[vllm]` 安装选项面向 CUDA。更多配置见[训练说明](docs/TRAINING.md)。

## 训练

设置当前任务的数据，以及初始或上一任务的 Hugging Face 模型路径：

```bash
export TRAIN_FILES=/path/to/productivity_train.parquet
export VAL_FILES=/path/to/productivity_test.parquet
export MODEL_PATH=/path/to/previous_task_hf_checkpoint

bash scripts/train_cgl.sh
```

使用 `DRY_RUN=1 bash scripts/train_cgl.sh` 查看命令。持续学习按任务依次执行：完成一个任务后，将 actor checkpoint 导出成 Hugging Face 模型，再作为下一任务的 `MODEL_PATH`。转换命令见[训练说明](docs/TRAINING.md#data-and-training)。

### 消融配置

| 设置 | 追加到训练命令的参数 |
| --- | --- |
| CGL | 启动脚本默认配置 |
| 梯度直接相加 | `actor_rollout_ref.actor.sft_grpo_loss_add_linear=True` |
| 固定监督权重 | `actor_rollout_ref.actor.sft_grpo_mu=False` |
| 仅 GRPO | `actor_rollout_ref.actor.sft_grpo_loss=False` |

## 复现与验证

```bash
pip install -r requirements-cgl-test.txt
pytest -q tests/cgl
```

CPU 测试覆盖路由、损失、梯度更新、奖励回调和配置。当前发布版的完整 NPU 训练复验仍待完成；初始 SFT、模型权重及完整评测流程不包含在本仓库中。详见[验证记录](docs/VALIDATION.md)和[实现改动](docs/MIGRATION.md)。

## 引用

使用 CGL 或 AndroidControl-CL 时，请引用论文：

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

## 致谢与许可

训练框架基于 [verl / HybridFlow](https://github.com/verl-project/verl)。代码采用 [Apache-2.0](LICENSE)，保留上游[版权声明](Notice.txt)；数据与模型的许可单独适用。
