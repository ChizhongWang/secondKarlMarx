#!/bin/bash
# 直接运行知识图谱增强LLM，不依赖API服务器

echo "启动知识图谱增强LLM（不依赖API服务器）..."

# 设置环境变量
export PYTHONPATH=$PYTHONPATH:/home/featurize/secondKarlMarx
export DMX_API_KEY="sk-ZhF2CPvVSjuqOZOk5n2fbancon1CTyxWzS1TtQx84JB5GQOY"

# 运行脚本，使用--no-api参数
cd /home/featurize/secondKarlMarx
python marx_kg/scripts/kg_enhanced_llm.py --no-api

echo "知识图谱增强LLM已退出"
