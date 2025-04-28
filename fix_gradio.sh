#!/bin/bash
# 修复gradio版本问题

# 卸载当前版本的gradio
pip uninstall -y gradio

# 安装LLaMA Factory兼容的gradio版本
pip install "gradio>=3.38.0,<4.0.0"

# 检查安装
python -c "import gradio; print('Gradio version:', gradio.__version__)"
