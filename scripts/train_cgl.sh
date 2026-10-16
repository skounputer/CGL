#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
: "${TRAIN_FILES:?Set TRAIN_FILES to the current task training parquet}"
: "${VAL_FILES:?Set VAL_FILES to an AndroidControl-CL validation parquet}"
: "${MODEL_PATH:?Set MODEL_PATH to the initial or previous-task Hugging Face checkpoint}"

# On Ascend, source the vendor environments before invoking this script, or
# supply their paths here. CUDA users can set DEVICE=cuda.
if [[ -n "${ASCEND_TOOLKIT_ENV:-}" ]]; then source "$ASCEND_TOOLKIT_ENV"; fi
if [[ -n "${ASCEND_ATB_ENV:-}" ]]; then source "$ASCEND_ATB_ENV"; fi
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export USE_OPTIMIZED_MODEL="${USE_OPTIMIZED_MODEL:-0}"
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-outputs/tensorboard}"

command=("${PYTHON_BIN:-python3}" -m verl.trainer.main_ppo
    algorithm.adv_estimator=grpo
    "data.train_files=$TRAIN_FILES"
    "data.val_files=$VAL_FILES"
    "data.train_batch_size=${TRAIN_BATCH_SIZE:-512}"
    data.max_prompt_length=2048
    data.max_response_length=256
    data.filter_overlong_prompts=True
    data.truncation=error
    data.image_key=images
    "actor_rollout_ref.model.path=$MODEL_PATH"
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.model.use_remove_padding=True
    "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-16}"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE:-2}"
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.01
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
    "actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE:-2}"
    actor_rollout_ref.actor.sft_grpo_loss=True
    actor_rollout_ref.actor.sft_grpo_loss_add_linear=False
    actor_rollout_ref.actor.sft_grpo_mu=True
    "actor_rollout_ref.actor.sft_grpo_routing=${CGL_ROUTING:-paper}"
    "actor_rollout_ref.rollout.name=${ENGINE:-vllm}"
    actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6
    actor_rollout_ref.rollout.enable_chunked_prefill=False
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.free_cache_engine=False
    actor_rollout_ref.rollout.n=8
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    algorithm.use_kl_in_reward=False
    custom_reward_function.path=verl/utils/reward_score/android_control.py
    custom_reward_function.name=compute_score
    reward_model.launch_reward_fn_async=False
    trainer.critic_warmup=0
    "trainer.logger=['console','tensorboard']"
    trainer.project_name=CGL
    "trainer.experiment_name=${EXPERIMENT_NAME:-cgl_androidcontrol}"
    "trainer.n_gpus_per_node=${N_GPUS:-8}"
    trainer.nnodes=1
    trainer.save_freq=10
    trainer.test_freq=100
    trainer.total_epochs=8
    "trainer.default_local_dir=${OUTPUT_DIR:-outputs/cgl_androidcontrol}"
    "trainer.device=${DEVICE:-npu}"
    trainer.val_before_train=False
)
command+=("$@")
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
else
    exec "${command[@]}"
fi
