"""
模型加载器 - 用于加载训练好的模型进行推理
"""

import os
import logging
import torch
from typing import Tuple, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# 配置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def load_model_for_inference(model_path: str) -> Tuple[Any, Any]:
    """
    加载模型用于推理
    
    Args:
        model_path: 模型路径
        
    Returns:
        model: 加载的模型
        tokenizer: 加载的分词器
    """
    logger.info(f"Loading model from {model_path}")
    
    # 检查是否是LoRA模型
    adapter_path = os.path.join(model_path, "adapter_model")
    is_lora = os.path.exists(adapter_path)
    
    if is_lora:
        logger.info("Detected LoRA model")
        # 加载LoRA配置
        peft_config = PeftConfig.from_pretrained(adapter_path)
        base_model_path = peft_config.base_model_name_or_path
        
        # 加载基础模型
        logger.info(f"Loading base model from {base_model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # 加载LoRA适配器
        logger.info(f"Loading LoRA adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        
        # 合并LoRA权重（可选）
        # model = model.merge_and_unload()
    else:
        logger.info("Loading full model")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 确保分词器有EOS和PAD token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 设置为评估模式
    model.eval()
    
    logger.info("Model loaded successfully")
    return model, tokenizer
