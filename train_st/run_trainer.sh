#!/bin/bash
LOGDIR=/home/zczlyf7/Overton-PPO/train_st/logs
mkdir -p "$LOGDIR"
exec > >(tee -i "$LOGDIR/finetune_hpo_$(date +%Y%m%d_%H%M%S).log") 2>&1
export HF_HOME=/scratch/zczlyf7/hf_cache
export CUDA_HOME=/apps/cuda/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TRITON_CACHE_DIR=/scratch/zczlyf7/triton_cache
export CUDA_VISIBLE_DEVICES=0  
export PYTHONUNBUFFERED=1      
export WANDB_MODE=offline

python trainer_tripetloss.py