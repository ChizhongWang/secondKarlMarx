"""
马克思恩格斯知识图谱增强LLM脚本
将微调的Qwen模型与GraphRAG知识图谱结合
"""
import os
import sys
import json
import logging
import requests
from pathlib import Path
import argparse

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
        logging.FileHandler("kg_enhanced_llm.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class KGEnhancedLLM:
    """知识图谱增强的LLM"""
    
    def __init__(self, llm_api_url="http://localhost:8000/v1/chat/completions", use_kg=True):
        """初始化
        
        Args:
            llm_api_url: LLM API的URL
            use_kg: 是否使用知识图谱增强
        """
        self.llm_api_url = llm_api_url
        self.use_kg = use_kg
        
        if self.use_kg:
            # 初始化知识图谱查询引擎
            self.initialize_kg_query()
        
    def initialize_kg_query(self):
        """初始化知识图谱查询引擎"""
        # 加载配置
        config_path = PROJECT_ROOT / "config" / "graphrag_config.yaml"
        logger.info(f"加载配置文件: {config_path}")
        config = load_config(PROJECT_ROOT, config_path)
        
        # 初始化查询
        logger.info("初始化GraphRAG查询引擎")
        self.query_engine = GraphRagQuery(config=config)
        logger.info("GraphRAG查询引擎初始化完成")
    
    def query_kg(self, query_text):
        """查询知识图谱
        
        Args:
            query_text: 查询文本
            
        Returns:
            str: 知识图谱查询结果
        """
        if not self.use_kg:
            return None
            
        try:
            logger.info(f"查询知识图谱: {query_text}")
            response = self.query_engine.query(query_text)
            logger.info(f"知识图谱查询结果: {response}")
            return response
        except Exception as e:
            logger.error(f"查询知识图谱时出错: {str(e)}", exc_info=True)
            return None
    
    def query_llm(self, messages):
        """查询LLM
        
        Args:
            messages: 消息列表
            
        Returns:
            str: LLM回复
        """
        try:
            logger.info(f"查询LLM: {messages}")
            
            payload = {
                "model": "Qwen2.5-7B-Instruct",
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1024
            }
            
            response = requests.post(
                self.llm_api_url,
                headers={"Content-Type": "application/json"},
                json=payload
            )
            
            if response.status_code != 200:
                logger.error(f"LLM API返回错误: {response.status_code} {response.text}")
                return f"LLM API错误: {response.status_code}"
                
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            logger.info(f"LLM回复: {content}")
            
            return content
        except Exception as e:
            logger.error(f"查询LLM时出错: {str(e)}", exc_info=True)
            return f"查询LLM时出错: {str(e)}"
    
    def answer(self, query_text):
        """回答问题
        
        Args:
            query_text: 用户问题
            
        Returns:
            str: 回答
        """
        # 构建系统提示
        system_prompt = "你是一个基于马克思主义理论的助手，专注于回答关于马克思、恩格斯及其著作的问题。"
        
        # 如果启用了知识图谱，查询知识图谱
        kg_result = None
        if self.use_kg:
            kg_result = self.query_kg(query_text)
        
        # 构建消息
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # 如果有知识图谱结果，添加到系统提示中
        if kg_result:
            kg_prompt = f"""我将为你提供一些相关的知识图谱信息，这些信息来自马克思恩格斯的著作。
请基于这些信息回答用户的问题，同时结合你自己的知识。
如果知识图谱信息与问题相关，请优先使用这些信息。

知识图谱信息:
{kg_result}

请回答以下问题，并明确指出你的回答中哪些部分是基于知识图谱的，哪些是基于你自己的知识的。"""
            
            messages.append({"role": "system", "content": kg_prompt})
        
        # 添加用户问题
        messages.append({"role": "user", "content": query_text})
        
        # 查询LLM
        return self.query_llm(messages)

def run_interactive_mode(llm_api_url, use_kg):
    """运行交互式模式"""
    kg_llm = KGEnhancedLLM(llm_api_url=llm_api_url, use_kg=use_kg)
    
    print("\n=== 马克思恩格斯知识图谱增强LLM ===")
    print(f"知识图谱增强: {'启用' if use_kg else '禁用'}")
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
            response = kg_llm.answer(user_query)
            
            print(f"\n回答: {response}")
            
        except KeyboardInterrupt:
            print("\n查询已中断")
            break
        except Exception as e:
            logger.error(f"查询时出错: {str(e)}", exc_info=True)
            print(f"\n查询时出错: {str(e)}")
    
    print("\n感谢使用马克思恩格斯知识图谱增强LLM！")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="马克思恩格斯知识图谱增强LLM")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000/v1/chat/completions",
                        help="LLM API的URL")
    parser.add_argument("--no-kg", action="store_true", help="禁用知识图谱增强")
    parser.add_argument("--query", type=str, help="单次查询的问题")
    
    args = parser.parse_args()
    
    if args.query:
        # 单次查询模式
        kg_llm = KGEnhancedLLM(llm_api_url=args.api_url, use_kg=not args.no_kg)
        response = kg_llm.answer(args.query)
        print(f"\n问题: {args.query}")
        print(f"\n回答: {response}")
    else:
        # 交互式模式
        run_interactive_mode(args.api_url, not args.no_kg)

if __name__ == "__main__":
    main()
