#!/bin/bash
# 运行集成了知识图谱的MCP服务器
# 这个脚本会启动一个集成了马克思恩格斯知识图谱的MCP服务器

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# 使用默认的模型配置文件
MODEL_CONFIG_PATH="$SCRIPT_DIR/../../mcp/model_config.json"

# 检查配置文件是否存在
if [ ! -f "$MODEL_CONFIG_PATH" ]; then
    echo "错误: 模型配置文件不存在: $MODEL_CONFIG_PATH"
    exit 1
fi

echo "使用模型配置文件: $MODEL_CONFIG_PATH"

# 启动MCP服务器
SCRIPT_PATH="$SCRIPT_DIR/mcp_server.py"
python $SCRIPT_PATH --config_path "$MODEL_CONFIG_PATH" --port 8080

# 如果需要指定不同的模型路径，可以使用以下命令
# python $SCRIPT_PATH --config_path "$MODEL_CONFIG_PATH" --model_path "你的模型路径" --port 8080
