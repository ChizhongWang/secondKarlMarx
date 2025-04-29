#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MCP服务器 - SSE传输版本
用于通过HTTP/SSE传输提供Qwen2.5-7B-Instruct微调模型的访问
"""

import logging
import torch
import json
import os
import time
from typing import List, Dict, Any, Optional
from mcp.server.fastmcp import FastMCP
from transformers import AutoModelForCausalLM, AutoTokenizer

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

# 全局变量存储模型和tokenizer
model = None
tokenizer = None
chat_history = []

# 加载模型函数
def load_model_for_inference():
    """加载LLaMA Factory微调模型"""
    logger.info(f"Loading model {MODEL_CONFIG['model_name_or_path']}...")
    
    # 加载原始模型
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_CONFIG["model_name_or_path"],
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CONFIG["model_name_or_path"],
        trust_remote_code=True
    )
    
    # 加载LoRA适配器（如果指定）
    if MODEL_CONFIG.get("adapter_name_or_path"):
        logger.info(f"Loading LoRA adapter from {MODEL_CONFIG['adapter_name_or_path']}...")
        from peft import PeftModel
        model = PeftModel.from_pretrained(
            model,
            MODEL_CONFIG["adapter_name_or_path"],
            torch_dtype=torch.float16
        )
    
    logger.info("Model loaded successfully!")
    return model, tokenizer

@karlmarx.tool()
def chat(message: str) -> str:
    """与Qwen2.5-7B-Instruct微调模型对话
    
    Args:
        message: 用户消息
    """
    # 使用全局模型和tokenizer
    global model, tokenizer, chat_history
    
    start_time = time.time()
    logger.info(f"收到聊天请求: {message[:50]}...")
    
    try:
        # 生成回复
        inputs = tokenizer(message, return_tensors="pt").to(model.device)
        
        # 记录tokenization时间
        tokenize_time = time.time() - start_time
        logger.info(f"Tokenization完成，耗时: {tokenize_time:.2f}秒")
        
        # 生成回复
        gen_start = time.time()
        outputs = model.generate(
            **inputs,
            max_new_tokens=MODEL_CONFIG.get("max_new_tokens", 512),  # 减少最大token数以加快响应
            temperature=MODEL_CONFIG.get("temperature", 0.7),
            top_p=MODEL_CONFIG.get("top_p", 0.9),
            repetition_penalty=MODEL_CONFIG.get("repetition_penalty", 1.1)
        )
        
        # 记录生成时间
        gen_time = time.time() - gen_start
        logger.info(f"生成完成，耗时: {gen_time:.2f}秒")
        
        # 解码回复
        decode_start = time.time()
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # 记录解码时间
        decode_time = time.time() - decode_start
        logger.info(f"解码完成，耗时: {decode_time:.2f}秒")
        
        # 记录总时间
        total_time = time.time() - start_time
        logger.info(f"总处理时间: {total_time:.2f}秒")
        
        # 保存对话历史
        chat_history.append({"user": message, "assistant": response})
        
        return response
    except Exception as e:
        logger.error(f"聊天过程中出错: {str(e)}")
        return f"抱歉，处理您的请求时出现错误: {str(e)}"

@karlmarx.tool()
def clear_history() -> str:
    """清除对话历史"""
    global chat_history
    logger.info("清除对话历史")
    chat_history = []
    return "对话历史已清除"

@karlmarx.tool()
def get_model_info() -> dict:
    """获取模型信息"""
    logger.info("获取模型信息")
    return {
        "model_name": MODEL_CONFIG["model_name_or_path"],
        "adapter": MODEL_CONFIG["adapter_name_or_path"],
        "finetuning_type": MODEL_CONFIG["finetuning_type"],
        "template": MODEL_CONFIG["template"]
    }

if __name__ == "__main__":
    # 加载模型
    model, tokenizer = load_model_for_inference()
    
    # 预热模型
    logger.info("预热模型...")
    dummy_input = "你好"
    _ = chat(dummy_input)
    logger.info("模型预热完成")
    
    # 运行MCP服务器 - 使用SSE传输
    logger.info(f"Starting MCP server for secondKarlMarx with SSE transport")
    # 使用SSE传输，不需要指定host和port
    karlmarx.run(transport='sse')
