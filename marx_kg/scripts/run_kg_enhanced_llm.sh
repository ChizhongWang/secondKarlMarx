#!/bin/bash
# 运行知识图谱增强LLM脚本

# 设置DMX API密钥
export DMX_API_KEY="sk-ZhF2CPvVSjuqOZOk5n2fbancon1CTyxWzS1TtQx84JB5GQOY"

# 运行脚本
python /home/featurize/secondKarlMarx/marx_kg/scripts/kg_enhanced_llm.py --api-url "http://localhost:8000/v1/chat/completions"
