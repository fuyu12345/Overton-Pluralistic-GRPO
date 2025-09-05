
LOGDIR=/p/project1/westai0052/self_rewarding/verl/output_logs
mkdir -p "$LOGDIR"
exec > >(tee -i "$LOGDIR/Summary-grpo-Qwen2.5-3B-Instruct-im-rewardscaledown-unique-12k$(date +%Y%m%d_%H%M%S).log") 2>&1

export RAY_TMPDIR="/tmp/ray_tmp"

module load GCC/13.3.0
export CC=$(which gcc)

export CUDA_HOME=/p/software/jurecadc/stages/2025/software/CUDA/12
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export CUDNN_HOME=~/libs/cudnn-9.8
export LD_LIBRARY_PATH=$CUDNN_HOME/lib:$LD_LIBRARY_PATH
export CPATH=$CUDNN_HOME/include:$CPATH



export CUDA_VISIBLE_DEVICES=0,1,2,3

export WANDB_MODE=offline
export WANDB_PROJECT=verl-Qwen2.5-3B-Instruct_im-rewardscaledown-unique-12k








PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
 algorithm.adv_estimator=grpo \
 data.train_files=/p/project1/westai0052/self_rewarding/verl/train_dataset/grpo_data_im_sum_12k/train.parquet \
 data.val_files=/p/project1/westai0052/self_rewarding/verl/train_dataset/grpo_data_im_sum_12k/test.parquet \
 reward_model.reward_manager=batch \
 custom_reward_function.path=/p/project1/westai0052/self_rewarding/verl/verl/utils/reward_score/op_reward_batch.py \
 data.train_batch_size=1024 \
 data.max_prompt_length=512 \
 data.max_response_length=800 \
 data.return_full_prompt=True \
 actor_rollout_ref.model.path=/p/project1/westai0052/self_rewarding/verl/output_model/sft_Qwen2.5-3B-Instruct_summary_full_1epoch-im/global_step_105 \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.model.use_remove_padding=True \
 actor_rollout_ref.actor.ppo_mini_batch_size=256 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=40 \
 actor_rollout_ref.actor.use_kl_loss=True \
 actor_rollout_ref.actor.kl_loss_coef=0.001 \
 actor_rollout_ref.actor.kl_loss_type=low_var_kl \
 actor_rollout_ref.actor.entropy_coeff=0 \
 actor_rollout_ref.model.enable_gradient_checkpointing=True \
 actor_rollout_ref.actor.fsdp_config.param_offload=False \
 actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
 actor_rollout_ref.rollout.n=10 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
 actor_rollout_ref.ref.fsdp_config.param_offload=True \
 trainer.logger=['console','wandb'] \
 algorithm.use_kl_in_reward=False \
 trainer.critic_warmup=0 \
 trainer.default_local_dir=/p/project1/westai0052/self_rewarding/verl/output_model/op_grpo/grpo-Qwen2.5-3B-Instruct-im-rewardscaledown-unique-12k \
 trainer.val_before_train=False \
 trainer.default_hdfs_dir=null \
 trainer.n_gpus_per_node=4 \
 trainer.nnodes=1 \
 trainer.save_freq=15 \
 trainer.test_freq=10 \
 trainer.total_epochs=15 2>&1 | tee verl_demo.log




#  PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
#  data.train_files=/p/project1/westai0052/self_rewarding/verl/train_dataset/train.parquet \
#  data.val_files=/p/project1/westai0052/self_rewarding/verl/train_dataset/test.parquet \
#  custom_reward_function.path=/p/project1/westai0052/self_rewarding/verl/verl/utils/reward_score/op_reward.py \
#  data.train_batch_size=256 \
#  data.max_prompt_length=512 \
#  data.max_response_length=512 \
#  data.return_full_prompt=True \
#  actor_rollout_ref.model.path=/p/project1/westai0052/self_rewarding/open-r1-main/HF_models_datasets/models/Qwen2.5-1.5B-Instruct \
#  actor_rollout_ref.actor.optim.lr=1e-6 \
#  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
#  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
#  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
#  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
#  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
#  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
#  critic.optim.lr=1e-5 \
#  critic.model.path=/p/project1/westai0052/self_rewarding/open-r1-main/HF_models_datasets/models/Qwen2.5-1.5B-Instruct \
#  critic.ppo_micro_batch_size_per_gpu=4 \
#  algorithm.kl_ctrl.kl_coef=0.001 \
#  trainer.logger=['console','wandb'] \
#  trainer.val_before_train=False \
#  trainer.default_hdfs_dir=null \
#  trainer.n_gpus_per_node=4 \
#  trainer.nnodes=1 \
#  trainer.save_freq=10 \
#  trainer.test_freq=10 \
#  trainer.total_epochs=15 2>&1 | tee verl_demo.log