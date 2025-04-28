#!/bin/bash
# LLaMA Factory训练启动脚本

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 使用4张GPU
export NCCL_DEBUG=INFO

# 设置内存优化
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 启动训练
python train_llama_factory.py \
    --model_name_or_path Qwen/Qwen2.5-7B \
    --dataset your_dataset \
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
    --fp16 