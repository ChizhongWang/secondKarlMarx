#!/bin/bash
# 安装LLaMA Factory及其依赖

# 卸载当前的llama-factory（如果已安装）
pip uninstall -y llama-factory

# 确保安装指定版本的依赖
pip install torch==2.6.0
pip install transformers==4.51.3

# 直接从GitHub安装LLaMA Factory
echo "从GitHub安装LLaMA Factory..."
pip install git+https://github.com/hiyouga/LLaMA-Factory.git

# 检查安装是否成功
if command -v llamafactory-cli &> /dev/null; then
    echo "LLaMA Factory安装成功！"
    echo "可以使用 'llamafactory-cli' 命令了"
else
    echo "LLaMA Factory安装可能失败，请检查错误信息"
fi
