"""
推理脚本 - 实现模型加载和推理功能
"""

import os
import sys
import logging
import torch
from typing import Dict, Optional, List, Union
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel, PeftConfig

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.training_config import BASE_MODEL_CONFIG

# 配置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(
    base_model_path: Optional[str] = None,
    lora_path: Optional[str] = None,
    device: Optional[str] = None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    加载模型和分词器，支持 LoRA 权重
    
    Args:
        base_model_path: 基础模型路径，如果为 None 则使用配置文件中的路径
        lora_path: LoRA 权重路径，如果为 None 则不加载 LoRA 权重
        device: 设备类型，如果为 None 则自动选择
    
    Returns:
        model: 加载的模型
        tokenizer: 加载的分词器
    """
    # 设置设备
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # 设置基础模型路径
    if base_model_path is None:
        base_model_path = BASE_MODEL_CONFIG["model_name_or_path"]
    logger.info(f"Loading base model from: {base_model_path}")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        padding_side="left",
    )
    
    # 确保分词器有正确的 EOS 和 PAD token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 创建4位量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    
    # 如果提供了 LoRA 路径，加载并合并 LoRA 权重
    if lora_path is not None and os.path.exists(lora_path):
        logger.info(f"Loading LoRA weights from: {lora_path}")
        try:
            # 加载 LoRA 配置
            peft_config = PeftConfig.from_pretrained(lora_path)
            logger.info(f"LoRA config: {peft_config}")
            
            # 加载并合并 LoRA 权重
            model = PeftModel.from_pretrained(
                model,
                lora_path,
                torch_dtype=torch.float16,
            )
            logger.info("LoRA weights loaded and merged successfully")
        except Exception as e:
            logger.error(f"Error loading LoRA weights: {e}")
            raise
    else:
        logger.info("No LoRA weights provided, using base model only")
    
    return model, tokenizer

def generate_response(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_length: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    num_return_sequences: int = 1,
) -> List[str]:
    """
    生成回复
    
    Args:
        model: 模型
        tokenizer: 分词器
        prompt: 输入提示
        max_length: 最大生成长度
        temperature: 温度参数
        top_p: top-p 采样参数
        top_k: top-k 采样参数
        repetition_penalty: 重复惩罚参数
        num_return_sequences: 返回的序列数量
    
    Returns:
        responses: 生成的回复列表
    """
    # 编码输入
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(model.device)
    
    # 生成回复
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # 解码输出
    responses = []
    for output in outputs:
        response = tokenizer.decode(output, skip_special_tokens=True)
        responses.append(response[len(prompt):])  # 移除输入提示
    
    return responses

def main():
    """主函数"""
    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer()
    
    # 测试生成
    prompt = "你好，请介绍一下你自己。"
    logger.info(f"Input prompt: {prompt}")
    
    responses = generate_response(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
    )
    
    for i, response in enumerate(responses, 1):
        logger.info(f"Response {i}: {response}")

if __name__ == "__main__":
    main() 