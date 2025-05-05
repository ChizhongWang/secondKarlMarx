#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import torch
import json
import os
import time
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from transformers import AutoModelForCausalLM, AutoTokenizer

# 配置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# 加载模型配置
MODEL_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../mcp/model_config.json")
with open(MODEL_CONFIG_PATH, "r") as f:
    MODEL_CONFIG = json.load(f)

# 创建FastAPI应用
app = FastAPI(title="Qwen API Server", description="OpenAI API compatible server for Qwen model")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量存储模型和tokenizer
model = None
tokenizer = None
chat_history = []

# 定义API请求和响应模型
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: float = Field(0.7, ge=0, le=2)
    top_p: float = Field(0.9, ge=0, le=1)
    max_tokens: int = Field(1024, ge=1)
    stream: bool = False

class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

class ModelListResponse(BaseModel):
    object: str
    data: List[Dict[str, Any]]

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

# 聊天函数
def generate_response(messages: List[ChatMessage]) -> str:
    """生成回复"""
    global model, tokenizer
    
    start_time = time.time()
    
    # 将消息格式化为模型输入
    template = MODEL_CONFIG.get("template", "qwen")
    
    # 构建提示
    if template == "qwen":
        prompt = ""
        for msg in messages:
            if msg.role == "system":
                prompt += f" system\n{msg.content} \n"
            elif msg.role == "user":
                prompt += f" user\n{msg.content} \n"
            elif msg.role == "assistant":
                prompt += f" assistant\n{msg.content} \n"
        
        # 添加最后的assistant标记
        prompt += " assistant\n"
    else:
        # 简单拼接
        prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in messages])
        prompt += "\nassistant: "
    
    logger.info(f"生成提示: {prompt[:100]}...")
    
    try:
        # 生成回复
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # 记录tokenization时间
        tokenize_time = time.time() - start_time
        logger.info(f"Tokenization完成，耗时: {tokenize_time:.2f}秒")
        
        # 生成回复
        gen_start = time.time()
        outputs = model.generate(
            **inputs,
            max_new_tokens=MODEL_CONFIG.get("max_new_tokens", 512),
            temperature=MODEL_CONFIG.get("temperature", 0.7),
            top_p=MODEL_CONFIG.get("top_p", 0.9),
            repetition_penalty=MODEL_CONFIG.get("repetition_penalty", 1.1)
        )
        
        # 记录生成时间
        gen_time = time.time() - gen_start
        logger.info(f"生成完成，耗时: {gen_time:.2f}秒")
        
        # 解码回复
        decode_start = time.time()
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取助手回复部分
        if template == "qwen":
            # 对于Qwen模板，提取最后一个assistant部分
            response = full_output.split(" assistant\n")[-1].split(" ")[0].strip()
        else:
            # 对于简单模板，提取assistant:之后的内容
            response = full_output.split("assistant: ")[-1].strip()
        
        # 记录解码时间
        decode_time = time.time() - decode_start
        logger.info(f"解码完成，耗时: {decode_time:.2f}秒")
        
        # 记录总时间
        total_time = time.time() - start_time
        logger.info(f"总处理时间: {total_time:.2f}秒")
        
        return response
    except Exception as e:
        logger.error(f"生成回复时出错: {str(e)}")
        return f"抱歉，处理您的请求时出现错误: {str(e)}"

# API端点
@app.get("/v1/models")
async def list_models():
    """列出可用模型"""
    return ModelListResponse(
        object="list",
        data=[
            {
                "id": "qwen2.5-7b-instruct",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "user"
            }
        ]
    )

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """聊天补全API"""
    global model, tokenizer
    
    # 确保模型已加载
    if model is None or tokenizer is None:
        model, tokenizer = load_model_for_inference()
    
    try:
        # 生成回复
        response_text = generate_response(request.messages)
        
        # 构建响应
        return ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time())}",
            object="chat.completion",
            created=int(time.time()),
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }
            ],
            usage={
                "prompt_tokens": 0,  # 简化处理，不计算实际token数
                "completion_tokens": 0,
                "total_tokens": 0
            }
        )
    except Exception as e:
        logger.error(f"处理聊天请求时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # 加载模型
    model, tokenizer = load_model_for_inference()
    
    # 预热模型
    logger.info("预热模型...")
    dummy_messages = [ChatMessage(role="user", content="你好")]
    _ = generate_response(dummy_messages)
    logger.info("模型预热完成")
    
    # 启动服务器
    logger.info("启动OpenAI兼容的API服务器...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
