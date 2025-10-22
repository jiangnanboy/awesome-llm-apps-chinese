import streamlit as st
import nest_asyncio
from io import BytesIO
from agno.agent import Agent
from agno.document.reader.pdf_reader import PDFReader
from agno.models.openai import OpenAIChat
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.embedder.openai import OpenAIEmbedder
from agno.vectordb.pgvector import PgVector, SearchType
from agno.storage.agent.postgres import PostgresAgentStorage

# 应用nest_asyncio以允许嵌套事件循环，这是在Streamlit中运行异步函数所必需的
nest_asyncio.apply()

# PostgreSQL数据库连接字符串
DB_URL = "postgresql+psycopg://ai:ai@localhost:5532/ai"


# 设置助手的函数，利用缓存提高资源效率
@st.cache_resource
def setup_assistant(api_key: str) -> Agent:
    """初始化并返回一个AI助手代理，并使用缓存提高效率。

    此函数使用OpenAI GPT-4o-mini模型设置AI助手代理，并为其配置知识库、存储和网络搜索工具。
    该助手设计为首先搜索其知识库，然后再查询互联网，提供清晰简洁的答案。

    参数:
        api_key (str): 访问OpenAI服务所需的API密钥。

    返回:
        Agent: 一个初始化的助手代理，配置有语言模型、知识库、存储和用于增强功能的附加工具。
    """
    llm = OpenAIChat(id="gpt-4o-mini", api_key=api_key)
    # 使用存储、知识库和工具设置助手
    return Agent(
        id="auto_rag_agent",  # 助手名称
        model=llm,  # 要使用的语言模型
        storage=PostgresAgentStorage(table_name="auto_rag_storage", db_url=DB_URL),
        knowledge_base=PDFUrlKnowledgeBase(
            vector_db=PgVector(
                db_url=DB_URL,
                collection="auto_rag_docs",
                embedder=OpenAIEmbedder(id="text-embedding-ada-002", dimensions=1536, api_key=api_key),
            ),
            num_documents=3,  # 要检索的文档数量
        ),
        tools=[DuckDuckGoTools()],  # 用于通过DuckDuckGo进行网络搜索的附加工具
        instructions=[
            "首先搜索你的知识库。",
            "如果未找到，搜索互联网。",
            "提供清晰简洁的答案。",
        ],
        show_tool_calls=True,  # 显示工具调用过程
        search_knowledge=True,  # 启用知识库搜索
        markdown=True,  # 支持Markdown格式输出
        debug_mode=True,  # 启用调试模式
    )


# 向知识库添加PDF文档的函数
def add_document(agent: Agent, file: BytesIO):
    """将PDF文档添加到代理的知识库中。

    此函数从类文件对象中读取PDF文档，并将其内容添加到指定代理的知识库中。如果文档成功读取，
    内容将加载到知识库中，并可选择更新现有数据。

    参数:
        agent (Agent): 其知识库将被更新的代理。
        file (BytesIO): 包含要添加的PDF文档的类文件对象。

    返回:
        None: 该函数不返回值，但会提供操作是否成功的反馈。
    """
    reader = PDFReader()
    docs = reader.read(file)
    if docs:
        agent.knowledge_base.load_documents(docs, upsert=True)
        st.success("文档已添加到知识库。")
    else:
        st.error("读取文档失败。")


# 查询助手并返回响应的函数
def query_assistant(agent: Agent, question: str) -> str:
    """查询助手并返回响应。

    参数:
        agent (Agent): 用于处理查询的Agent类实例。
        question (str): 要向助手提出的问题。

    返回:
        str: 助手针对给定问题生成的响应。
    """
    return "".join([delta for delta in agent.run(question)])


# 处理Streamlit应用布局和交互的主函数
def main():
    """处理Streamlit应用布局和交互的主函数。

    此函数设置Streamlit应用配置，处理用户输入（如OpenAI API密钥、PDF上传和用户问题），
    并与基于GPT-4o的自主检索增强生成（RAG）助手进行交互。

    该应用允许用户上传PDF文档以增强知识库，并提交问题以接收生成的响应。

    副作用:
        - 配置Streamlit页面和标题。
        - 提示用户输入OpenAI API密钥和问题。
        - 允许用户上传PDF文档。
        - 显示通过查询助手生成的响应。

    异常:
        StreamlitWarning: 如果未提供OpenAI API密钥。
    """
    st.set_page_config(page_title="AutoRAG", layout="wide")
    st.title("🤖 Auto-RAG: 基于GPT-4o的自主检索增强生成")

    api_key = st.sidebar.text_input("输入你的OpenAI API密钥 🔑", type="password")

    if not api_key:
        st.sidebar.warning("请输入你的OpenAI API密钥以继续。")
        st.stop()

    assistant = setup_assistant(api_key)

    uploaded_file = st.sidebar.file_uploader("📄 上传PDF", type=["pdf"])

    if uploaded_file and st.sidebar.button("🛠️ 添加到知识库"):
        add_document(assistant, BytesIO(uploaded_file.read()))

    question = st.text_input("💬 提出你的问题:")

    # 当用户提交问题时，向助手查询答案
    if st.button("🔍 获取答案"):
        # 确保问题不为空
        if question.strip():
            with st.spinner("🤔 思考中..."):
                # 查询助手并显示响应
                answer = query_assistant(assistant, question)
                st.write("📝 **响应:**", answer.content)
        else:
            # 如果问题输入为空，显示错误
            st.error("请输入一个问题。")


# 应用程序的入口点
if __name__ == "__main__":
    main()