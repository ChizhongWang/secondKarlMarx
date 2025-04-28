#!/bin/bash
# LLaMA Factory训练启动脚本

# 默认使用所有可用GPU
NUM_GPUS=${1:-"all"}

# 设置环境变量
if [ "$NUM_GPUS" = "all" ]; then
    # 使用所有可用GPU
    unset CUDA_VISIBLE_DEVICES
    echo "使用所有可用GPU"
else
    # 构建GPU ID列表 (0,1,2,...,NUM_GPUS-1)
    GPU_LIST=$(seq -s, 0 $((NUM_GPUS-1)))
    export CUDA_VISIBLE_DEVICES=$GPU_LIST
    echo "使用GPU: $GPU_LIST"
fi

export NCCL_DEBUG=INFO

# 设置内存优化
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 启动训练
python train_llama_factory.py \
    --model_name_or_path Qwen/Qwen2.5-7B \
    --dataset custom \
    --dataset_dir ./training \
    --template qwen \
    --finetuning_type lora \
    --lora_target q_proj,k_proj,v_proj,o_proj \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --cutoff_len 8192 \
    --output_dir ./outputs_llama_factory \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 100 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --fp16 \
    --deepspeed configs/ds_config_zero2.json