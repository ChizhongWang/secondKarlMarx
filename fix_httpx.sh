#!/bin/bash
# 安装httpx的socks扩展

# 在zsh中，方括号需要加引号
pip install --user "httpx[socks]"

# 安装其他可能缺少的依赖
pip install --user gradio

# 检查安装
python -c "import httpx; print('httpx version:', httpx.__version__)"
