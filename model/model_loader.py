"""
模型加载器 - 负责加载预训练模型和 LoRA 权重
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import logging

logger = logging.getLogger(__name__)

def load_model_and_tokenizer(model_path, lora_path=None, device="cuda"):
    """
    加载模型和分词器
    
    Args:
        model_path: 基础模型路径
        lora_path: LoRA 权重路径
        device: 设备类型
    
    Returns:
        model: 加载的模型
        tokenizer: 加载的分词器
    """
    try:
        # 加载分词器
        logger.info(f"Loading tokenizer from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left",
            truncation_side="left"
        )
        
        # 加载基础模型
        logger.info(f"Loading base model from {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
            use_cache=True
        )
        
        # 如果有 LoRA 权重，加载并合并
        if lora_path and os.path.exists(lora_path):
            logger.info(f"Loading LoRA weights from {lora_path}")
            try:
                # 加载 LoRA 配置
                peft_config = PeftConfig.from_pretrained(lora_path)
                logger.info(f"LoRA config: {peft_config}")
                
                # 加载 LoRA 模型
                model = PeftModel.from_pretrained(
                    model,
                    lora_path,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                
                # 合并 LoRA 权重到基础模型
                logger.info("Merging LoRA weights into base model")
                model = model.merge_and_unload()
                
            except Exception as e:
                logger.error(f"Error loading LoRA weights: {str(e)}")
                raise
        
        # 将模型移动到指定设备
        if device != "auto":
            model = model.to(device)
        
        logger.info("Model and tokenizer loaded successfully")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise
