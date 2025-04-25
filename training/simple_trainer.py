"""
简化版训练器 - 不依赖TRL库，使用基本的Transformers Trainer
"""

import os
import logging
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from configs.training_config import (
    BASE_MODEL_CONFIG,
    DATASET_CONFIG,
    TRAINING_CONFIG,
    LORA_CONFIG,
)
from training.data_utils import load_sft_dataset

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
    
    # 创建4位量化配置
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    # 加载模型 - 使用4位量化以节省内存
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_CONFIG["model_name_or_path"],
        quantization_config=quantization_config,
        device_map="auto",  # 自动分配到可用设备
        trust_remote_code=BASE_MODEL_CONFIG["trust_remote_code"],
        token=BASE_MODEL_CONFIG["use_auth_token"],
        use_cache=False,  # 训练时禁用KV缓存以节省内存
        # 注意：张量并行现在由环境中的PyTorch 2.5+自动处理
    )
    
    # 应用LoRA适配器进行参数高效微调
    logger.info("Applying LoRA adapter for parameter-efficient fine-tuning")
    
    # 为量化训练准备模型
    model = prepare_model_for_kbit_training(model)
    
    # 根据模型类型选择目标模块
    target_modules = None
    model_type = model.config.model_type.lower() if hasattr(model.config, "model_type") else ""
    
    # 为Qwen2.5模型设置特定的目标模块
    if "qwen" in model_type:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "w1", "w2", "w3"]
        logger.info(f"Using Qwen-specific target modules: {target_modules}")
    else:
        # 默认LoRA目标模块
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        logger.info(f"Using default target modules: {target_modules}")
    
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
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def train_simple(local_rank=None):
    """
    简化版训练函数 - 使用基本的Transformers Trainer
    
    Args:
        local_rank: 本地排名，用于分布式训练
    
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
    
    # 创建数据收集器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # 不使用掩码语言建模
    )
    
    # 创建训练参数
    training_args_dict = TRAINING_CONFIG.copy()
    
    # 确保使用正确的分布式设置
    training_args_dict["ddp_backend"] = "nccl"  # 使用NCCL后端以提高多GPU性能
    
    training_args = TrainingArguments(
        **training_args_dict,
        remove_unused_columns=False,  # 保留所有列
    )
    
    # 创建基本Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
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
