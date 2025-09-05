set -x
LOGDIR=/p/project1/westai0052/self_rewarding/verl/output_logs
mkdir -p "$LOGDIR"
exec > >(tee -i "$LOGDIR/SFT_Llama-3.2-3B-Instruct-summary-im$(date +%Y%m%d_%H%M%S).log") 2>&1

export RAY_TMPDIR="/tmp/ray_tmp"

module load GCC/13.3.0
export CC=$(which gcc)

export CUDA_HOME=/p/software/jurecadc/stages/2025/software/CUDA/12
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export CUDNN_HOME=~/libs/cudnn-9.8
export LD_LIBRARY_PATH=$CUDNN_HOME/lib:$LD_LIBRARY_PATH
export CPATH=$CUDNN_HOME/include:$CPATH


export WANDB_MODE=offline
export WANDB_PROJECT=sft_Llama-3.2-3B-Instruct_summary


export CUDA_VISIBLE_DEVICES=0,1,2,3



torchrun --nproc_per_node=4 -m verl.trainer.fsdp_sft_trainer \
    data.train_files=/p/project1/westai0052/self_rewarding/verl/train_dataset/sft_ex_sum_full_implicit/train.parquet \
    data.val_files=/p/project1/westai0052/self_rewarding/verl/train_dataset/sft_ex_sum_full_implicit/test.parquet \
    data.prompt_key=flattened_prompt \
    data.response_key=flattened_response \
    data.max_length=1300 \
    data.micro_batch_size_per_gpu=8 \
    model.partial_pretrain=/p/project1/westai0052/self_rewarding/open-r1-main/HF_models_datasets/models/Llama-3.2-3B-Instruct \
    trainer.project_name=Llama-3.2-3B-Instruct-Instruct-OPV2-SFT-summary-im \
    trainer.experiment_name=Llama-3.2-3B-Instruct-OPV2-SFT-summary-im \
    trainer.total_epochs=1 \
    trainer.logger=['console'] \
    trainer.default_local_dir=/p/project1/westai0052/self_rewarding/verl/output_model/sft_Llama-3.2-3B-Instruct_summary_full_1epoch-im \
