"""
MCP服务器 - 为训练好的Qwen2.5-7B-Instruct LoRA微调模型提供MCP接口
"""

import os
import sys
import logging
import torch
import json
from typing import List, Dict, Any, Optional
from mcp.server.fastmcp import FastMCP
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 配置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# 加载模型配置
MODEL_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "model_config.json")
with open(MODEL_CONFIG_PATH, "r") as f:
    MODEL_CONFIG = json.load(f)

# 创建MCP服务
karlmarx = FastMCP("secondKarlMarx")

# 加载模型函数
def load_model_for_inference():
    """加载LLaMA Factory微调模型"""
    logger.info(f"Loading model {MODEL_CONFIG['model_name_or_path']}...")
    
    # 配置量化参数（如果GPU内存不足）
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,  # 使用8位量化以节省内存
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # 加载原始模型
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_CONFIG["model_name_or_path"],
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=quantization_config
    )
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CONFIG["model_name_or_path"],
        trust_remote_code=True
    )
    
    # 如果有LoRA适配器，加载它
    if MODEL_CONFIG.get("adapter_name_or_path"):
        logger.info(f"Loading LoRA adapter from {MODEL_CONFIG['adapter_name_or_path']}...")
        from peft import PeftModel
        model = PeftModel.from_pretrained(
            model, 
            MODEL_CONFIG["adapter_name_or_path"]
        )
    
    logger.info("Model loaded successfully!")
    return model, tokenizer

# 加载模型
model, tokenizer = load_model_for_inference()

# 消息历史
messages = []

@karlmarx.tool()
async def chat(query: str) -> str:
    """
    与微调后的Qwen2.5模型进行对话
    
    Args:
        query (str): 用户的问题或指令
    
    Returns:
        str: 模型的回答
    """
    global messages
    
    try:
        # 添加用户消息
        messages.append({"role": "user", "content": query})
        
        # 准备输入
        inputs = tokenizer.apply_chat_template(
            messages, 
            return_tensors="pt",
            add_generation_prompt=True
        ).to(model.device)
        
        # 生成回答
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=MODEL_CONFIG.get("max_new_tokens", 1024),
                temperature=MODEL_CONFIG.get("temperature", 0.7),
                top_p=MODEL_CONFIG.get("top_p", 0.9),
                repetition_penalty=MODEL_CONFIG.get("repetition_penalty", 1.1),
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 解码回答
        response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        
        # 添加助手消息
        messages.append({"role": "assistant", "content": response})
        
        # 如果消息历史太长，保留最近的10条
        if len(messages) > 20:
            messages = messages[-20:]
        
        return response
    
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        return f"抱歉，处理您的请求时出现错误: {str(e)}"

@karlmarx.tool()
async def clear_history() -> str:
    """
    清除对话历史
    
    Returns:
        str: 操作结果
    """
    global messages
    messages = []
    return "对话历史已清除"

@karlmarx.tool()
async def get_model_info() -> Dict[str, Any]:
    """
    获取模型信息
    
    Returns:
        Dict[str, Any]: 模型信息
    """
    return {
        "model_name": MODEL_CONFIG["model_name_or_path"],
        "adapter": MODEL_CONFIG.get("adapter_name_or_path", "None"),
        "template": MODEL_CONFIG.get("template", "qwen"),
        "finetuning_type": MODEL_CONFIG.get("finetuning_type", "lora")
    }

if __name__ == "__main__":
    # 初始化并运行服务器
    logger.info(f"Starting MCP server for secondKarlMarx")
    karlmarx.run(transport='stdio')
