"""
训练脚本 - 实现分布式训练功能
"""

import os
import sys
import logging
import torch
from typing import Dict, Optional
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import deepspeed

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.data_utils import load_sft_dataset
from configs.training_config import (
    BASE_MODEL_CONFIG,
    DATASET_CONFIG,
    TRAINING_CONFIG,
    LORA_CONFIG,
    DEEPSPEED_CONFIG,
)

# 配置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def setup_model_and_tokenizer():
    """
    设置模型和分词器
    
    Returns:
        model: 加载的模型
        tokenizer: 加载的分词器
    """
    logger.info(f"Loading base model: {BASE_MODEL_CONFIG['model_name_or_path']}")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_CONFIG["model_name_or_path"],
        use_fast=True,
        trust_remote_code=BASE_MODEL_CONFIG["trust_remote_code"],
        token=BASE_MODEL_CONFIG["use_auth_token"],
        padding_side="right",  # Qwen模型使用right padding
    )
    
    # 确保分词器有正确的EOS和PAD token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型 - 使用4位量化以节省内存
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_CONFIG["model_name_or_path"],
        load_in_4bit=True,  # 使用4位量化
        device_map="auto",  # 自动分配到可用设备
        quantization_config=transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
        trust_remote_code=BASE_MODEL_CONFIG["trust_remote_code"],
        token=BASE_MODEL_CONFIG["use_auth_token"],
    )
    
    # 为量化训练准备模型
    model = prepare_model_for_kbit_training(model)
    
    # 应用LoRA - 为Qwen模型调整target_modules
    logger.info("Applying LoRA adapters")
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    # Qwen2.5特定的target_modules
    if "Qwen" in BASE_MODEL_CONFIG["model_name_or_path"]:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "w1", "w2", "w3"]
    
    lora_config = LoraConfig(
        r=LORA_CONFIG["r"],
        lora_alpha=LORA_CONFIG["lora_alpha"],
        lora_dropout=LORA_CONFIG["lora_dropout"],
        bias=LORA_CONFIG["bias"],
        task_type=LORA_CONFIG["task_type"],
        target_modules=target_modules,
        fan_in_fan_out=LORA_CONFIG.get("fan_in_fan_out", False),
    )
    model = get_peft_model(model, lora_config)
    
    # 打印模型可训练参数数量
    model.print_trainable_parameters()
    
    return model, tokenizer

def train(local_rank=None, deepspeed_config=None, zero_stage=None):
    """
    训练模型
    
    Args:
        local_rank: 本地排名，用于分布式训练
        deepspeed_config: DeepSpeed配置文件路径
        zero_stage: ZeRO优化阶段（覆盖配置文件中的设置）
    
    Returns:
        final_model_path: 最终模型保存路径
    """
    # 设置随机种子
    set_seed(TRAINING_CONFIG.get("seed", 42))
    
    # 设置本地排名
    if local_rank is not None:
        os.environ["LOCAL_RANK"] = str(local_rank)
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_main_process = local_rank == -1 or local_rank == 0
    
    # 加载模型和分词器
    model, tokenizer = setup_model_and_tokenizer()
    
    # 加载数据集
    logger.info("Loading dataset")
    train_dataset, eval_dataset = load_sft_dataset(
        dataset_name=DATASET_CONFIG["dataset_name"],
        dataset_config_name=DATASET_CONFIG.get("dataset_config_name"),
        split=DATASET_CONFIG.get("split"),
        streaming=DATASET_CONFIG.get("streaming", False),
        max_samples=DATASET_CONFIG.get("max_samples"),
        format=DATASET_CONFIG.get("format", "prompt_content"),
        prompt_field=DATASET_CONFIG.get("prompt_field", "prompt"),
        content_field=DATASET_CONFIG.get("content_field", "content"),
        preprocessing_num_workers=8,
        tokenizer=tokenizer,
        max_seq_length=TRAINING_CONFIG.get("max_seq_length", 8192),
    )
    
    # 创建训练参数
    training_args_dict = TRAINING_CONFIG.copy()
    
    # 处理DeepSpeed配置
    if deepspeed_config:
        logger.info(f"Using DeepSpeed config from: {deepspeed_config}")
        training_args_dict["deepspeed"] = deepspeed_config
    
    # 如果指定了ZeRO阶段，覆盖配置
    if zero_stage is not None:
        logger.info(f"Overriding ZeRO stage to: {zero_stage}")
        if "deepspeed" not in training_args_dict:
            training_args_dict["deepspeed"] = {}
        if isinstance(training_args_dict["deepspeed"], str):
            # 如果deepspeed是字符串（文件路径），我们不能直接修改
            logger.warning("Cannot override ZeRO stage when deepspeed config is a file path")
        else:
            training_args_dict["deepspeed"]["zero_optimization"] = {"stage": zero_stage}
    
    training_args = TrainingArguments(
        **training_args_dict,
        remove_unused_columns=False,  # 保留所有列
        label_names=["labels"],
        logging_first_step=True,
        ddp_backend="nccl",  # 使用NCCL后端以提高多GPU性能
    )
    
    # 创建SFT训练器
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        packing=True,  # 启用序列打包以提高效率
        max_seq_length=TRAINING_CONFIG.get("max_seq_length", 2048),
        dataset_text_field=DATASET_CONFIG.get("text_field", "text"),
    )
    
    # 开始训练
    logger.info("Starting training")
    trainer.train()
    
    # 保存最终模型
    if is_main_process:
        final_model_path = os.path.join(TRAINING_CONFIG["output_dir"], "final_model")
        logger.info(f"Saving final model to {final_model_path}")
        trainer.save_model(final_model_path)
        tokenizer.save_pretrained(final_model_path)
    
    return TRAINING_CONFIG["output_dir"]

if __name__ == "__main__":
    train()
