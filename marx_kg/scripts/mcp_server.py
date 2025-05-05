"""
MCP服务器集成GraphRAG知识图谱
为本地笔记本提供远程访问secondKarlMarx和马克思恩格斯知识图谱的能力
"""
import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent
GRAPHRAG_ROOT = PROJECT_ROOT.parent / "graphrag"
sys.path.append(str(GRAPHRAG_ROOT))
sys.path.append(str(PROJECT_ROOT.parent))  # 添加secondKarlMarx根目录

# 加载模型配置
MODEL_CONFIG_PATH = Path(PROJECT_ROOT.parent) / "mcp" / "model_config.json"
if MODEL_CONFIG_PATH.exists():
    with open(MODEL_CONFIG_PATH, "r") as f:
        MODEL_CONFIG = json.load(f)
    logger.info(f"已加载模型配置: {MODEL_CONFIG_PATH}")
else:
    MODEL_CONFIG = {}
    logger.warning(f"未找到模型配置文件: {MODEL_CONFIG_PATH}")

# 导入kg_tool
from marx_kg.scripts.kg_tool import query_marx_kg

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mcp_server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 尝试导入MCP相关模块
try:
    from mcp.server import MCPServer
    from mcp.models import ChatMessage, ChatCompletionRequest, ChatCompletionResponse, ToolCall, Tool
    HAS_MCP = True
except ImportError:
    logger.warning("未找到MCP模块，将使用模拟实现")
    HAS_MCP = False
    # 定义模拟的MCP类
    class MCPServer:
        def __init__(self, *args, **kwargs):
            pass
        def run(self, *args, **kwargs):
            logger.error("MCP模块未安装，无法启动服务器")
            sys.exit(1)
    
    class ChatMessage:
        def __init__(self, role, content):
            self.role = role
            self.content = content
    
    class ChatCompletionRequest:
        def __init__(self, messages, model, tools=None):
            self.messages = messages
            self.model = model
            self.tools = tools
    
    class ChatCompletionResponse:
        def __init__(self, choices):
            self.choices = choices
    
    class ToolCall:
        def __init__(self, id, name, arguments):
            self.id = id
            self.name = name
            self.arguments = arguments
    
    class Tool:
        def __init__(self, name, description, parameters):
            self.name = name
            self.description = description
            self.parameters = parameters

# 定义知识图谱查询工具
KG_TOOL = {
    "name": "query_marx_kg",
    "description": "查询马克思恩格斯知识图谱，获取相关信息",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "要查询的问题或关键词"
            }
        },
        "required": ["query"]
    }
}

class SecondKarlMarxMCPServer(MCPServer):
    """
    扩展MCP服务器，集成马克思恩格斯知识图谱查询功能
    """
    
    def __init__(self, model_path: str = None, host: str = "0.0.0.0", port: int = 8000):
        """
        初始化服务器
        
        Args:
            model_path: secondKarlMarx模型路径
            host: 服务器主机地址
            port: 服务器端口
        """
        # 如果未指定模型路径，使用配置文件中的路径
        if model_path is None and MODEL_CONFIG:
            model_path = MODEL_CONFIG.get("model_name_or_path")
            logger.info(f"使用配置文件中的模型路径: {model_path}")
            
        super().__init__(model_path=model_path, host=host, port=port)
        self.tools = [Tool(**KG_TOOL)]
        logger.info(f"已加载知识图谱查询工具: {KG_TOOL['name']}")
        
        # 如果配置文件中有adapter路径，加载adapter
        if MODEL_CONFIG and MODEL_CONFIG.get("adapter_name_or_path"):
            self.adapter_path = MODEL_CONFIG.get("adapter_name_or_path")
            logger.info(f"使用配置文件中的adapter路径: {self.adapter_path}")
        else:
            self.adapter_path = None
    
    def handle_tool_calls(self, tool_calls: List[ToolCall]) -> str:
        """
        处理工具调用
        
        Args:
            tool_calls: 工具调用列表
            
        Returns:
            工具调用结果
        """
        results = []
        
        for tool_call in tool_calls:
            if tool_call.name == "query_marx_kg":
                try:
                    args = json.loads(tool_call.arguments)
                    query = args.get("query", "")
                    
                    if not query:
                        results.append({
                            "tool_call_id": tool_call.id,
                            "result": json.dumps({"error": "查询参数不能为空"})
                        })
                        continue
                    
                    # 调用知识图谱查询
                    kg_result = query_marx_kg(query)
                    
                    results.append({
                        "tool_call_id": tool_call.id,
                        "result": json.dumps(kg_result, ensure_ascii=False)
                    })
                    
                except Exception as e:
                    logger.error(f"处理工具调用时出错: {str(e)}", exc_info=True)
                    results.append({
                        "tool_call_id": tool_call.id,
                        "result": json.dumps({"error": f"处理工具调用时出错: {str(e)}"})
                    })
            else:
                # 调用父类处理其他工具
                return super().handle_tool_calls(tool_calls)
        
        return results
    
    def process_request(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """
        处理聊天请求
        
        Args:
            request: 聊天请求
            
        Returns:
            聊天响应
        """
        # 添加工具到请求
        if not hasattr(request, 'tools') or not request.tools:
            request.tools = self.tools
        
        # 调用父类处理请求
        return super().process_request(request)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="SecondKarlMarx MCP服务器")
    parser.add_argument("--model_path", type=str, required=False, help="secondKarlMarx模型路径")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")
    parser.add_argument("--config_path", type=str, default=None, help="模型配置文件路径")
    args = parser.parse_args()
    
    # 如果指定了配置文件路径，加载该配置
    if args.config_path:
        global MODEL_CONFIG
        with open(args.config_path, "r") as f:
            MODEL_CONFIG = json.load(f)
        logger.info(f"从命令行指定的路径加载模型配置: {args.config_path}")
    
    if not HAS_MCP:
        logger.error("未找到MCP模块，请先安装MCP")
        sys.exit(1)
    
    # 启动服务器
    server = SecondKarlMarxMCPServer(
        model_path=args.model_path,
        host=args.host,
        port=args.port
    )
    
    logger.info(f"启动SecondKarlMarx MCP服务器，地址: {args.host}:{args.port}")
    server.run()

if __name__ == "__main__":
    main()
