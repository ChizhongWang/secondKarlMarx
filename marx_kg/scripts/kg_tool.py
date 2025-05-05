"""
知识图谱查询工具
为secondKarlMarx提供查询马克思恩格斯知识图谱的能力
"""
import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent
GRAPHRAG_ROOT = PROJECT_ROOT.parent / "graphrag"
sys.path.append(str(GRAPHRAG_ROOT))

# 导入GraphRAG
from graphrag.config.load_config import load_config
from graphrag.query.graph_rag_query import GraphRagQuery

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("kg_tool.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 全局查询引擎实例
_query_engine = None

def get_query_engine():
    """获取或初始化查询引擎"""
    global _query_engine
    
    if _query_engine is None:
        # 加载配置
        config_path = PROJECT_ROOT / "config" / "graphrag_config.yaml"
        logger.info(f"加载配置文件: {config_path}")
        config = load_config(PROJECT_ROOT, config_path)
        
        # 初始化查询
        logger.info("初始化GraphRAG查询引擎")
        _query_engine = GraphRagQuery(config=config)
    
    return _query_engine

def query_marx_kg(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    查询马克思恩格斯知识图谱
    
    Args:
        query: 查询文本
        max_results: 最大返回结果数
        
    Returns:
        包含查询结果的字典
    """
    try:
        logger.info(f"知识图谱查询: {query}")
        
        # 获取查询引擎
        query_engine = get_query_engine()
        
        # 执行查询
        response = query_engine.query(query)
        
        # 获取相关实体和关系（如果可用）
        context = query_engine.get_last_context() if hasattr(query_engine, 'get_last_context') else None
        
        # 构建结果
        result = {
            "answer": response,
            "source": "马克思恩格斯知识图谱",
            "entities": [],
            "relations": []
        }
        
        # 如果有上下文信息，添加到结果中
        if context and isinstance(context, dict):
            if 'entities' in context:
                result["entities"] = context['entities'][:max_results] if isinstance(context['entities'], list) else []
            if 'relations' in context:
                result["relations"] = context['relations'][:max_results] if isinstance(context['relations'], list) else []
        
        logger.info(f"查询成功: 找到 {len(result['entities'])} 个实体和 {len(result['relations'])} 个关系")
        return result
        
    except Exception as e:
        logger.error(f"查询时出错: {str(e)}", exc_info=True)
        return {
            "answer": f"查询知识图谱时出错: {str(e)}",
            "source": "马克思恩格斯知识图谱",
            "entities": [],
            "relations": []
        }

# 工具函数定义（用于与secondKarlMarx集成）
def kg_tool(query: str) -> str:
    """
    知识图谱查询工具函数
    
    Args:
        query: 查询文本
        
    Returns:
        查询结果的JSON字符串
    """
    result = query_marx_kg(query)
    return json.dumps(result, ensure_ascii=False)

# 测试函数
def test_query(query: str):
    """测试知识图谱查询"""
    result = query_marx_kg(query)
    print(f"\n查询: {query}")
    print(f"\n回答: {result['answer']}")
    
    if result['entities']:
        print("\n相关实体:")
        for entity in result['entities']:
            print(f"  - {entity.get('text', '')}: {entity.get('type', '')}")
    
    if result['relations']:
        print("\n相关关系:")
        for relation in result['relations']:
            print(f"  - {relation.get('head_text', '')} {relation.get('relation', '')} {relation.get('tail_text', '')}")

if __name__ == "__main__":
    # 如果直接运行此脚本，执行测试查询
    if len(sys.argv) > 1:
        test_query(' '.join(sys.argv[1:]))
    else:
        test_query("马克思关于资本主义的观点是什么？")
