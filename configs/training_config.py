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
    "max_samples": 100,  # 增加样本数量用于实际训练
    "streaming": False,
}

# 训练配置
TRAINING_CONFIG = {
    "output_dir": "./outputs",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 1,  # 减小批处理大小以减少内存使用
    "gradient_accumulation_steps": 8,  # 调整梯度累积步数
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "logging_steps": 5,
    "save_strategy": "steps",
    "save_steps": 100,
    "save_total_limit": 3,
    "fp16": True,
    "bf16": False,
    "max_grad_norm": 1.0,
    "max_seq_length": 1536,  # 减小序列长度以减轻内存压力
    "max_steps": 100,  # 限制最大步数为100
    "eval_steps": 50,
    "group_by_length": True,
    "report_to": "none",  # 移除wandb以简化测试
    "ddp_find_unused_parameters": False,
    "dataloader_num_workers": 7,  # 增加数据加载器工作线程数量以适应7张GPU
    "dataloader_pin_memory": True,
    "local_rank": -1,
    "remove_unused_columns": True,
}

# LoRA配置
LORA_CONFIG = {
    "r": 32,  # 降低LoRA秩以减少参数量
    "lora_alpha": 64,  # LoRA alpha参数，通常设置为2*r
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
    "fan_in_fan_out": False,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # 指定LoRA目标模块
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
