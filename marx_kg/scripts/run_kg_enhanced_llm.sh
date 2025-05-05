#!/bin/bash

# 设置环境变量
export DMX_API_KEY="sk-ZhF2CPvVSjuqOZOk5n2fbancon1CTyxWzS1TtQx84JB5GQOY"

# 运行知识图谱增强LLM脚本
python marx_kg/scripts/kg_enhanced_llm.py "$@"
