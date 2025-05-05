"""
马克思恩格斯知识图谱查询脚本
使用GraphRAG查询知识图谱
"""
import os
import sys
import logging
from pathlib import Path

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
        logging.FileHandler("query_kg.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def initialize_query():
    """初始化查询引擎"""
    # 加载配置
    config_path = PROJECT_ROOT / "config" / "graphrag_config.yaml"
    logger.info(f"加载配置文件: {config_path}")
    config = load_config(PROJECT_ROOT, config_path)
    
    # 初始化查询
    logger.info("初始化GraphRAG查询引擎")
    query = GraphRagQuery(config=config)
    
    return query

def run_interactive_query():
    """运行交互式查询"""
    query_engine = initialize_query()
    
    print("\n=== 马克思恩格斯知识图谱查询系统 ===")
    print("输入您的问题，或输入'exit'退出\n")
    
    while True:
        try:
            user_query = input("\n问题: ")
            if user_query.lower() in ['exit', 'quit', '退出']:
                break
                
            if not user_query.strip():
                continue
                
            logger.info(f"用户查询: {user_query}")
            
            # 执行查询
            response = query_engine.query(user_query)
            
            print(f"\n回答: {response}")
            
        except KeyboardInterrupt:
            print("\n查询已中断")
            break
        except Exception as e:
            logger.error(f"查询时出错: {str(e)}", exc_info=True)
            print(f"\n查询时出错: {str(e)}")
    
    print("\n感谢使用马克思恩格斯知识图谱查询系统！")

def run_single_query(query_text):
    """运行单个查询"""
    query_engine = initialize_query()
    
    try:
        logger.info(f"执行查询: {query_text}")
        response = query_engine.query(query_text)
        print(f"\n问题: {query_text}")
        print(f"\n回答: {response}")
        return response
    except Exception as e:
        logger.error(f"查询时出错: {str(e)}", exc_info=True)
        print(f"\n查询时出错: {str(e)}")
        return None

def main():
    """主函数"""
    if len(sys.argv) > 1:
        # 如果提供了命令行参数，将其作为查询
        query_text = ' '.join(sys.argv[1:])
        run_single_query(query_text)
    else:
        # 否则运行交互式查询
        run_interactive_query()

if __name__ == "__main__":
    main()
