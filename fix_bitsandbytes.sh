#!/bin/bash
# 修复bitsandbytes与CUDA 12.6的兼容性问题

echo "安装兼容CUDA 12.6的bitsandbytes版本..."
pip uninstall -y bitsandbytes
pip install -U --no-build-isolation git+https://github.com/TimDettmers/bitsandbytes.git

# 创建符号链接以解决库文件问题
echo "创建符号链接..."
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
cd $SITE_PACKAGES/bitsandbytes
if [ -f "libbitsandbytes_cuda126.so" ]; then
    ln -sf libbitsandbytes_cuda126.so libbitsandbytes_cpu.so
elif [ -f "libbitsandbytes_cuda120.so" ]; then
    ln -sf libbitsandbytes_cuda120.so libbitsandbytes_cpu.so
fi

echo "完成！"
