# CGL

[论文](https://openaccess.thecvf.com/content/CVPR2026/html/Yao_CGL_Advancing_Continual_GUI_Learning_via_Reinforcement_Fine-Tuning_CVPR_2026_paper.html)：**CGL: Advancing Continual GUI Learning via Reinforcement Fine-Tuning**。

[English README](README.md) · [AndroidControl-CL 数据集](https://huggingface.co/datasets/Regimes/androidcontrol-cl)

这是根据研究代码片段整理的首版开源代码，包含完整的历史 verl 底座、CGL 算法接入、AndroidControl 奖励函数、训练入口和 CPU 测试。**当前整理版本尚未经过完整 GPU／Ascend 训练复验，不能视为论文数值结果已复现。** 初始 SFT 流水线、模型权重、完整评测程序和权威任务顺序文件尚未包含。

底座固定为 2025-07-07 的 verl 提交 `3f929af747777f43041b8c6c56da26c909842e27`，版本文件是 `0.4.1.dev`。这是与所给文件高度匹配的重建底座；缺少原始 Git 历史，因此无法证明原实验 checkout 的 HEAD 正好是这个提交。本版没有迁移到最新 verl。

## 训练入口

原始实验使用八张 Ascend NPU、vLLM 0.7.0 和 Qwen-VL 类模型。安装时使用配套的 CANN、PyTorch/torch-npu 和 vllm-ascend 环境，并按 `requirements-npu.txt` 安装项目依赖。环境安装、数据字段和可调整参数见 [英文说明](README.md)。

```bash
export TRAIN_FILES=/你的数据目录/productivity_train.parquet
export VAL_FILES=/你的数据目录/productivity_test.parquet
export MODEL_PATH=/你的模型目录/上一任务的HF格式模型

# 仅查看命令。
DRY_RUN=1 bash scripts/train_cgl.sh

# 在已配置的 Ascend 环境中训练一个任务。
bash scripts/train_cgl.sh
```

一个任务训练完成后，需要将 actor checkpoint 导出成 Hugging Face 模型，再作为下一任务的 `MODEL_PATH`。不要直接把 FSDP 分片目录当作 HF 模型路径。数据和权重不进入 Git。

## 方法开关

| 配置 | 含义 |
| --- | --- |
| `sft_grpo_loss=True` | 启用纠正回答的监督分支 |
| `sft_grpo_mu=True` | 使用原代码中的熵权重调度 |
| `sft_grpo_loss_add_linear=False` | 对冲突梯度做投影；True 为直接相加消融 |
| `sft_grpo_routing=paper` | 同一 prompt 的所有采样都未达到满分 2 时触发监督 |

接入范围是 FSDP actor、vanilla policy loss、同步奖励。保留原实现的逐参数／局部分片投影，并不等价于对全模型分布式梯度做一次全局投影。FSDP2 等继承后端还需硬件验证。

整理过程中修复了奖励回调接口不匹配、首列奖励索引、batch 重排后的分组、纠正序列位置与 padding、截断示范、动态 microbatch 样本计数，以及关闭 SFT 时误清空策略梯度的问题。默认路由与论文一致；可用 `CGL_ROUTING=legacy_nonzero` 保留旧的“非零奖励”阈值，但不会恢复原索引错误。详见 [迁移记录](docs/MIGRATION.md)。这些变化可能影响旧实验的训练轨迹。

## CPU 验证

```bash
pip install -r requirements-cgl-test.txt
pytest -q tests/cgl
python -m compileall -q verl
bash -n scripts/train_cgl.sh
```

CPU 测试不替代 GPU／NPU、分布式和完整训练验证。引用信息见 [CITATION.cff](CITATION.cff)，代码采用 [Apache-2.0](LICENSE)，保留上游版权和声明；数据与模型的许可单独适用。
