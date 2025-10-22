import os
from typing import List, Dict, Any, Literal, Optional
from dataclasses import dataclass
import streamlit as st
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import tempfile
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from langchain.schema import HumanMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain import hub
from langgraph.prebuilt import create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.language_models import BaseLanguageModel
from langchain.prompts import ChatPromptTemplate
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


def init_session_state():
    """初始化会话状态变量"""
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = ""
    if 'qdrant_url' not in st.session_state:
        st.session_state.qdrant_url = ""
    if 'qdrant_api_key' not in st.session_state:
        st.session_state.qdrant_api_key = ""
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = None
    if 'llm' not in st.session_state:
        st.session_state.llm = None
    if 'databases' not in st.session_state:
        st.session_state.databases = {}


init_session_state()

DatabaseType = Literal["products", "support", "finance"]
PERSIST_DIRECTORY = "db_storage"


@dataclass
class CollectionConfig:
    name: str
    description: str
    collection_name: str  # 这将用作Qdrant集合名称


# 集合配置
COLLECTIONS: Dict[DatabaseType, CollectionConfig] = {
    "products": CollectionConfig(
        name="产品信息",
        description="产品详情、规格和特性",
        collection_name="products_collection"
    ),
    "support": CollectionConfig(
        name="客户支持与FAQ",
        description="客户支持信息、常见问题和指南",
        collection_name="support_collection"
    ),
    "finance": CollectionConfig(
        name="财务信息",
        description="财务数据、收入、成本和负债",
        collection_name="finance_collection"
    )
}


def initialize_models():
    """初始化OpenAI模型和Qdrant客户端"""
    if (st.session_state.openai_api_key and
            st.session_state.qdrant_url and
            st.session_state.qdrant_api_key):

        os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key
        st.session_state.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        st.session_state.llm = ChatOpenAI(temperature=0)

        try:
            client = QdrantClient(
                url=st.session_state.qdrant_url,
                api_key=st.session_state.qdrant_api_key
            )

            # 测试连接
            client.get_collections()
            vector_size = 1536  # text-embedding-3-small的向量大小
            st.session_state.databases = {}
            for db_type, config in COLLECTIONS.items():
                try:
                    client.get_collection(config.collection_name)
                except Exception:
                    # 如果集合不存在则创建
                    client.create_collection(
                        collection_name=config.collection_name,
                        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
                    )

                st.session_state.databases[db_type] = Qdrant(
                    client=client,
                    collection_name=config.collection_name,
                    embeddings=st.session_state.embeddings
                )

            return True
        except Exception as e:
            st.error(f"连接Qdrant失败: {str(e)}")
            return False
    return False


def process_document(file) -> List[Document]:
    """处理上传的PDF文档"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        # 清理临时文件
        os.unlink(tmp_path)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)

        return texts
    except Exception as e:
        st.error(f"处理文档时出错: {e}")
        return []


def create_routing_agent() -> Agent:
    """使用agno框架创建路由代理"""
    return Agent(
        model=OpenAIChat(
            id="gpt-4o",
            api_key=st.session_state.openai_api_key
        ),
        tools=[],
        description="""你是一名查询路由专家。你的唯一工作是分析问题并确定它们应该被路由到哪个数据库。
        你必须准确地返回以下三个选项之一：'products'、'support'或'finance'。用户的问题是：{question}""",
        instructions=[
            "严格遵守以下规则：",
            "1. 关于产品、功能、规格、物品细节或产品手册的问题 → 返回'products'",
            "2. 关于帮助、指导、故障排除、客户服务、FAQ或指南的问题 → 返回'support'",
            "3. 关于成本、收入、定价、财务数据或财务报告和投资的问题 → 返回'finance'",
            "4. 只返回数据库名称，不要其他文本或解释",
            "5. 如果你对路由不确定，返回空响应"
        ],
        markdown=False,
        show_tool_calls=False
    )


def route_query(question: str) -> Optional[DatabaseType]:
    """通过搜索所有数据库并比较相关性分数来路由查询。
    如果没有找到合适的数据库，返回None。"""
    try:
        best_score = -1
        best_db_type = None
        all_scores = {}  # 存储所有分数用于调试

        # 搜索每个数据库并比较相关性分数
        for db_type, db in st.session_state.databases.items():
            results = db.similarity_search_with_score(
                question,
                k=3
            )

            if results:
                avg_score = sum(score for _, score in results) / len(results)
                all_scores[db_type] = avg_score

                if avg_score > best_score:
                    best_score = avg_score
                    best_db_type = db_type

        confidence_threshold = 0.5
        if best_score >= confidence_threshold and best_db_type:
            st.success(f"使用向量相似度路由: {best_db_type} (置信度: {best_score:.3f})")
            return best_db_type

        st.warning(f"低置信度分数(低于{confidence_threshold})，退回到LLM路由")

        # 退回到LLM路由
        routing_agent = create_routing_agent()
        response = routing_agent.run(question)

        db_type = (response.content
                   .strip()
                   .lower()
                   .translate(str.maketrans('', '', '`\'"')))

        if db_type in COLLECTIONS:
            st.success(f"使用LLM路由决策: {db_type}")
            return db_type

        st.warning("未找到合适的数据库，将使用网络搜索作为后备")
        return None

    except Exception as e:
        st.error(f"路由错误: {str(e)}")
        return None


def create_fallback_agent(chat_model: BaseLanguageModel):
    """创建用于网络研究的LangGraph代理。"""

    def web_research(query: str) -> str:
        """带结果格式化的网络搜索。"""
        try:
            search = DuckDuckGoSearchRun(num_results=5)
            results = search.run(query)
            return results
        except Exception as e:
            return f"搜索失败: {str(e)}。基于一般知识提供答案。"

    tools = [web_research]

    agent = create_react_agent(model=chat_model,
                               tools=tools,
                               debug=False)

    return agent


def query_database(db: Qdrant, question: str) -> tuple[str, list]:
    """查询数据库并返回答案和相关文档"""
    try:
        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

        relevant_docs = retriever.get_relevant_documents(question)

        if relevant_docs:
            # 使用更简单的链创建方式和hub提示
            retrieval_qa_prompt = ChatPromptTemplate.from_messages([
                ("system", """你是一个有帮助的AI助手，根据提供的上下文回答问题。
                             回答时始终保持直接和简洁。
                             如果上下文没有包含足够的信息来完全回答问题，请承认这一限制。
                             严格基于提供的上下文回答，避免做出假设。"""),
                ("human", "以下是上下文:\n{context}"),
                ("human", "问题: {input}"),
                ("assistant", "我将根据提供的上下文帮助回答你的问题。"),
                ("human", "请提供你的答案:"),
            ])
            combine_docs_chain = create_stuff_documents_chain(st.session_state.llm, retrieval_qa_prompt)
            retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

            response = retrieval_chain.invoke({"input": question})
            return response['answer'], relevant_docs

        raise ValueError("在数据库中未找到相关文档")

    except Exception as e:
        st.error(f"错误: {str(e)}")
        return "我遇到了一个错误。请尝试重新表述你的问题。", []


def _handle_web_fallback(question: str) -> tuple[str, list]:
    st.info("未找到相关文档。正在搜索网络...")
    fallback_agent = create_fallback_agent(st.session_state.llm)

    with st.spinner('研究中...'):
        agent_input = {
            "messages": [
                HumanMessage(content=f"研究并为以下内容提供详细答案: '{question}'")
            ],
            "is_last_step": False
        }

        try:
            response = fallback_agent.invoke(agent_input, config={"recursion_limit": 100})
            if isinstance(response, dict) and "messages" in response:
                answer = response["messages"][-1].content
                return f"网络搜索结果:\n{answer}", []

        except Exception:
            # 退回到一般LLM响应
            fallback_response = st.session_state.llm.invoke(question).content
            return f"网络搜索不可用。一般响应: {fallback_response}", []


def main():
    """主应用函数。"""
    st.set_page_config(page_title="带数据库路由的RAG代理", page_icon="📚")
    st.title("📠 带数据库路由的RAG代理")

    # 用于API密钥和配置的侧边栏
    with st.sidebar:
        st.header("配置")

        # OpenAI API密钥
        api_key = st.text_input(
            "输入OpenAI API密钥:",
            type="password",
            value=st.session_state.openai_api_key,
            key="api_key_input"
        )

        # Qdrant配置
        qdrant_url = st.text_input(
            "输入Qdrant URL:",
            value=st.session_state.qdrant_url,
            help="示例: https://your-cluster.qdrant.tech"
        )

        qdrant_api_key = st.text_input(
            "输入Qdrant API密钥:",
            type="password",
            value=st.session_state.qdrant_api_key
        )

        # 更新会话状态
        if api_key:
            st.session_state.openai_api_key = api_key
        if qdrant_url:
            st.session_state.qdrant_url = qdrant_url
        if qdrant_api_key:
            st.session_state.qdrant_api_key = qdrant_api_key

        # 如果提供了所有凭据，则初始化模型
        if (st.session_state.openai_api_key and
                st.session_state.qdrant_url and
                st.session_state.qdrant_api_key):
            if initialize_models():
                st.success("成功连接到OpenAI和Qdrant!")
            else:
                st.error("初始化失败。请检查你的凭据。")
        else:
            st.warning("请输入所有必要的凭据以继续")
            st.stop()

        st.markdown("---")

    st.header("文档上传")
    st.info("上传文档以填充数据库。每个标签对应不同的数据库。")
    tabs = st.tabs([collection_config.name for collection_config in COLLECTIONS.values()])

    for (collection_type, collection_config), tab in zip(COLLECTIONS.items(), tabs):
        with tab:
            st.write(collection_config.description)
            uploaded_files = st.file_uploader(
                f"上传PDF文档到{collection_config.name}",
                type="pdf",
                key=f"upload_{collection_type}",
                accept_multiple_files=True  # 允许上传多个文件
            )

            if uploaded_files:
                with st.spinner('处理文档中...'):
                    all_texts = []
                    for uploaded_file in uploaded_files:
                        texts = process_document(uploaded_file)
                        all_texts.extend(texts)

                    if all_texts:
                        db = st.session_state.databases[collection_type]
                        db.add_documents(all_texts)
                        st.success("文档已处理并添加到数据库!")

    # 查询部分
    st.header("提问")
    st.info("在下方输入你的问题，从相关数据库中查找答案。")
    question = st.text_input("输入你的问题:")

    if question:
        with st.spinner('寻找答案中...'):
            # 路由问题
            collection_type = route_query(question)

            if collection_type is None:
                # 直接使用网络搜索后备
                answer, relevant_docs = _handle_web_fallback(question)
                st.write("### 答案(来自网络搜索)")
                st.write(answer)
            else:
                # 显示路由信息并查询数据库
                st.info(f"将问题路由到: {COLLECTIONS[collection_type].name}")
                db = st.session_state.databases[collection_type]
                answer, relevant_docs = query_database(db, question)
                st.write("### 答案")
                st.write(answer)


if __name__ == "__main__":
    main()