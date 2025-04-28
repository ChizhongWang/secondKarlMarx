"""
训练配置文件 - 定义训练参数和模型配置
"""

# 基础模型配置
BASE_MODEL_CONFIG = {
    "model_name_or_path": "Qwen/Qwen2.5-7B-Instruct"
}

# 训练数据集配置
DATASET_CONFIG = {
    "dataset_name": "ChizhongWang/secondKarlMarx-sft",
    "prompt_field": "prompt",  # 提示字段
    "content_field": "content",  # 内容字段
    "format": "prompt_content",  # 使用自定义格式
    "max_samples": 2,  # 只使用两条数据进行快速验证
    "streaming": False,
}

# 训练配置
TRAINING_CONFIG = {
    "output_dir": "./outputs",  # 确保输出目录正确
    "num_train_epochs": 1,  # 减少训练轮数
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 1,  # 减少梯度累积步数
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "logging_steps": 1,  # 每步都记录日志
    "save_strategy": "steps",
    "save_steps": 1,  # 每步都保存
    "save_total_limit": 1,  # 只保留一个检查点
    "fp16": True,
    "bf16": False,
    "max_grad_norm": 1.0,
    "max_seq_length": 1536,
    "max_steps": 2,  # 限制最大步数为2
    "eval_steps": 1,  # 每步都评估
    "group_by_length": True,
    "report_to": "none",
    "ddp_find_unused_parameters": False,
    "dataloader_num_workers": 1,  # 减少工作线程数
    "dataloader_pin_memory": True,
    "local_rank": -1,
    "remove_unused_columns": True,
}

# LoRA配置
LORA_CONFIG = {
    "r": 8,  # 降低LoRA秩以减少参数量
    "lora_alpha": 16,  # LoRA alpha参数，通常设置为2*r
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
    "fan_in_fan_out": False,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}

# DeepSpeed配置
DEEPSPEED_CONFIG = {
    "config_path": "./configs/ds_config.json",
    "stage": 3,
    "offload_optimizer": True,
    "offload_param": True,
    "overlap_comm": True,
    "pin_memory": True,
}

# MCP配置
MCP_CONFIG = {
    "server_name": "secondKarlMarx",
    "host": "0.0.0.0",
    "port": 8000,
    "model_path": "./outputs/final_model",  # 修改为与TRAINING_CONFIG一致的路径
}
