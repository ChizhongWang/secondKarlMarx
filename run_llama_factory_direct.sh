#!/bin/bash
# 直接使用LLaMA Factory的主入口点进行训练

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
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 创建配置文件
cat > train_config.json << EOF
{
    "model_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
    "dataset": "ChizhongWang/secondKarlMarx-sft",
    "prompt_column": "prompt",
    "response_column": "content",
    "max_samples": 2,
    "template": "qwen",
    "finetuning_type": "lora",
    "lora_target": "q_proj,k_proj,v_proj,o_proj",
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "cutoff_len": 8192,
    "output_dir": "./outputs_llama_factory_test",
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "lr_scheduler_type": "cosine",
    "logging_steps": 1,
    "save_steps": 100,
    "learning_rate": 2e-5,
    "num_train_epochs": 1.0,
    "fp16": true,
    "deepspeed": "configs/ds_config_zero2.json"
}
EOF

# 尝试多种可能的入口点
echo "尝试运行LLaMA Factory..."

# 方法1: 尝试使用python -m llmtuner
echo "尝试方法1: python -m llmtuner"
python -m llmtuner --train --config train_config.json || true

# 方法2: 尝试使用llmtuner命令
echo "尝试方法2: llmtuner命令"
llmtuner --train --config train_config.json || true

# 方法3: 尝试直接运行LLaMA Factory的主脚本
echo "尝试方法3: 直接运行主脚本"
cd "$(pip show llmtuner | grep Location | cut -d' ' -f2)/llmtuner" || true
python train.py --config ../../../train_config.json || true

echo "所有方法尝试完毕"
