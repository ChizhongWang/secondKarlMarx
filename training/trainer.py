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
    # 获取本地排名
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    logger.info(f"Setting up model on rank {local_rank}")
    
    # 设置当前设备
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        logger.info(f"Using GPU {local_rank}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    
    logger.info(f"Loading base model: {BASE_MODEL_CONFIG['model_name_or_path']}")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_CONFIG["model_name_or_path"], trust_remote_code=True)
    
    # 确保分词器有正确的EOS和PAD token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 创建4位量化配置
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # 加载模型 - 使用4位量化以减少内存使用
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_CONFIG["model_name_or_path"],
        quantization_config=bnb_config,
        device_map={"": local_rank},  # 回退到指定设备，避免DTensor问题
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    
    # 应用LoRA适配器进行参数高效微调
    logger.info("Applying LoRA adapter for parameter-efficient fine-tuning")
    
    # 为量化训练准备模型 - 添加错误处理
    try:
        model = prepare_model_for_kbit_training(model)
    except Exception as e:
        logger.warning(f"Error in prepare_model_for_kbit_training: {e}")
        logger.info("Proceeding without prepare_model_for_kbit_training")
    
    # 为Qwen2.5模型设置特定的目标模块
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    logger.info(f"Using target modules: {target_modules}")
    
    # 创建LoRA配置
    peft_config = LoraConfig(
        r=LORA_CONFIG["r"],
        lora_alpha=LORA_CONFIG["lora_alpha"],
        target_modules=target_modules,
        lora_dropout=LORA_CONFIG["lora_dropout"],
        bias=LORA_CONFIG["bias"],
        task_type=LORA_CONFIG["task_type"],
    )
    
    # 应用LoRA适配器
    try:
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    except Exception as e:
        logger.error(f"Error applying LoRA adapter: {e}")
        # 尝试备用方法
        logger.info("Trying alternative approach for LoRA application")
        from peft.tuners.lora import LoraModel
        model = LoraModel(model, peft_config, "default")
        logger.info("LoRA adapter applied using alternative method")
    
    return model, tokenizer

def train(local_rank=None, deepspeed_config=None, zero_stage=None):
    """
    训练模型
    
    Args:
        local_rank: 本地排名，用于分布式训练
        deepspeed_config: DeepSpeed配置
        zero_stage: ZeRO优化阶段
    
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
    
    # 从训练参数中移除 max_seq_length，因为它应该在 SFTTrainer 中使用
    if "max_seq_length" in training_args_dict:
        del training_args_dict["max_seq_length"]
    
    # 处理DeepSpeed配置
    if deepspeed_config:
        logger.info("Using provided DeepSpeed configuration")
        # 如果提供了deepspeed_config，使用它
        training_args_dict["deepspeed"] = deepspeed_config
    elif "deepspeed" in training_args_dict:
        # 否则使用配置文件中的默认设置
        logger.info("Using default DeepSpeed configuration from training config")
    else:
        # 如果没有DeepSpeed配置，移除相关设置
        logger.info("No DeepSpeed configuration provided, using PyTorch native distributed")
        if "deepspeed" in training_args_dict:
            del training_args_dict["deepspeed"]
    
    # 设置ZeRO阶段（如果提供）
    if zero_stage is not None:
        logger.info(f"Setting ZeRO stage to {zero_stage}")
        if "deepspeed" not in training_args_dict:
            training_args_dict["deepspeed"] = {}
        if isinstance(training_args_dict["deepspeed"], dict) and "zero_optimization" not in training_args_dict["deepspeed"]:
            training_args_dict["deepspeed"]["zero_optimization"] = {}
        if isinstance(training_args_dict["deepspeed"], dict):
            training_args_dict["deepspeed"]["zero_optimization"]["stage"] = zero_stage
    
    # 确保使用正确的分布式设置
    training_args_dict["ddp_backend"] = "nccl"  # 使用NCCL后端以提高多GPU性能
    
    training_args = TrainingArguments(
        **training_args_dict,
        remove_unused_columns=False,  # 保留所有列
        label_names=["labels"],
        logging_first_step=True,
    )
    
    # 创建SFT训练器 - 只使用核心参数，避免API兼容性问题
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    try:
        logger.info("Starting training")
        # 禁用默认的checkpoint保存，改为手动保存
        original_save_strategy = training_args.save_strategy
        training_args.save_strategy = "no"
        
        # 训练模型
        trainer.train()
        
        # 手动保存最终模型
        logger.info("Training completed, saving final model")
        # 使用PEFT的保存方法而不是DeepSpeed的方法
        if hasattr(model, "save_pretrained"):
            logger.info("Saving using PEFT save_pretrained")
            # 只保存LoRA适配器权重
            model.save_pretrained(training_args.output_dir)
        else:
            logger.warning("Model doesn't have save_pretrained method, using trainer.save_model")
            try:
                # 尝试使用trainer的保存方法，但可能会失败
                trainer.save_model(training_args.output_dir)
            except Exception as e:
                logger.error(f"Error saving model: {e}")
                logger.info("Saving tokenizer only as fallback")
                tokenizer.save_pretrained(training_args.output_dir)
                
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    
    return training_args.output_dir

if __name__ == "__main__":
    train()
