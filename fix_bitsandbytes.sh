#!/bin/bash
# 修复bitsandbytes与CUDA 12.3的兼容性问题

echo "安装兼容CUDA 12.3的bitsandbytes版本..."
pip uninstall -y bitsandbytes
pip install -U --no-build-isolation git+https://github.com/TimDettmers/bitsandbytes.git

echo "完成！"
