"""
训练配置文件 - 定义训练参数和模型配置
"""

# 基础模型配置
BASE_MODEL_CONFIG = {
    "model_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
    "trust_remote_code": True,
    "use_auth_token": True,
}

# 训练数据集配置
DATASET_CONFIG = {
    "dataset_name": "ChizhongWang/secondKarlMarx-sft",
    "text_field": "text",  # 根据您的数据集结构调整
    "format": "messages",  # 可选: "instruction", "messages", "text"
    "max_samples": None,  # 设置为None使用全部数据
    "streaming": False,
}

# 训练配置
TRAINING_CONFIG = {
    "output_dir": "./results",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,  # 减小批处理大小以适应更长的序列
    "gradient_accumulation_steps": 8,  # 增加梯度累积步数以保持有效批量大小
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.03,
    "logging_steps": 10,
    "save_steps": 500,
    "save_total_limit": 3,
    "fp16": True,
    "bf16": False,  # 3090不支持BF16
    "torch_compile": False,  # 可能导致内存问题，暂时禁用
    "gradient_checkpointing": True,  # 启用梯度检查点以节省内存
    "optim": "adamw_torch",
    "seed": 42,
    "max_seq_length": 4096,  # 增加到4096以处理更长的文本
    "max_steps": -1,  # 使用epoch而不是steps
    "save_strategy": "steps",
    "evaluation_strategy": "steps",
    "eval_steps": 500,
    "group_by_length": True,  # 按长度分组以提高效率
    "report_to": ["tensorboard", "wandb"],
    "ddp_find_unused_parameters": False,  # 提高DDP效率
    "dataloader_num_workers": 4,  # 数据加载器的工作线程数
    "dataloader_pin_memory": True,  # 使用固定内存提高数据传输速度
}

# LoRA配置
LORA_CONFIG = {
    "r": 64,  # LoRA秩，更高的值可能提高性能但增加参数量
    "lora_alpha": 128,  # LoRA alpha参数，通常设置为2*r
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
    "fan_in_fan_out": False,
}

# DeepSpeed配置
DEEPSPEED_CONFIG = {
    "config_path": "./configs/ds_config.json",
    "stage": 3,  # 使用ZeRO-3以处理更大模型
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
    "model_path": "./results/final_model",
}
