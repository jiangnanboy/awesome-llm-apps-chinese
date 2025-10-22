import os
import tempfile
from datetime import datetime
from typing import List
import streamlit as st
import bs4
from agno.agent import Agent
from agno.models.ollama import Ollama
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_core.embeddings import Embeddings
from agno.tools.exa import ExaTools
from agno.embedder.ollama import OllamaEmbedder


class OllamaEmbedderr(Embeddings):
    def __init__(self, model_name="snowflake-arctic-embed"):
        """
        使用特定模型初始化OllamaEmbedderr。

        参数:
            model_name (str): 用于嵌入的模型名称。
        """
        self.embedder = OllamaEmbedder(id=model_name, dimensions=1024)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self.embedder.get_embedding(text)


# 常量
COLLECTION_NAME = "test-deepseek-r1"

# Streamlit应用初始化
st.title("🐋 Deepseek本地RAG推理代理")

# 会话状态初始化
if 'google_api_key' not in st.session_state:
    st.session_state.google_api_key = ""
if 'qdrant_api_key' not in st.session_state:
    st.session_state.qdrant_api_key = ""
if 'qdrant_url' not in st.session_state:
    st.session_state.qdrant_url = ""
if 'model_version' not in st.session_state:
    st.session_state.model_version = "deepseek-r1:1.5b"  # 默认使用轻量模型
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'processed_documents' not in st.session_state:
    st.session_state.processed_documents = []
if 'history' not in st.session_state:
    st.session_state.history = []
if 'exa_api_key' not in st.session_state:
    st.session_state.exa_api_key = ""
if 'use_web_search' not in st.session_state:
    st.session_state.use_web_search = False
if 'force_web_search' not in st.session_state:
    st.session_state.force_web_search = False
if 'similarity_threshold' not in st.session_state:
    st.session_state.similarity_threshold = 0.7
if 'rag_enabled' not in st.session_state:
    st.session_state.rag_enabled = True  # 默认启用RAG

# 侧边栏配置
st.sidebar.header("🤖 代理配置")

# 模型选择
st.sidebar.header("📦 模型选择")
model_help = """
- 1.5b: 轻量模型，适合大多数笔记本电脑
- 7b: 能力更强但需要更好的GPU/内存

根据您的硬件性能选择。
"""
st.session_state.model_version = st.sidebar.radio(
    "选择模型版本",
    options=["deepseek-r1:1.5b", "deepseek-r1:7b"],
    help=model_help
)
st.sidebar.info("分别运行ollama pull deepseek-r1:7b或deepseek-r1:1.5b拉取模型")

# RAG模式切换
st.sidebar.header("🔍 RAG配置")
st.session_state.rag_enabled = st.sidebar.toggle("启用RAG模式", value=st.session_state.rag_enabled)

# 清除聊天按钮
if st.sidebar.button("🗑️ 清除聊天历史"):
    st.session_state.history = []
    st.rerun()

# 仅当RAG启用时显示API配置
if st.session_state.rag_enabled:
    st.sidebar.header("🔑 API配置")
    qdrant_api_key = st.sidebar.text_input("Qdrant API密钥", type="password", value=st.session_state.qdrant_api_key)
    qdrant_url = st.sidebar.text_input("Qdrant URL",
                                       placeholder="https://your-cluster.cloud.qdrant.io:6333",
                                       value=st.session_state.qdrant_url)

    # 更新会话状态
    st.session_state.qdrant_api_key = qdrant_api_key
    st.session_state.qdrant_url = qdrant_url

    # 搜索配置（仅在RAG模式下显示）
    st.sidebar.header("🎯 搜索配置")
    st.session_state.similarity_threshold = st.sidebar.slider(
        "文档相似度阈值",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        help="较低的值将返回更多文档但相关性可能较低。较高的值更严格。"
    )

# 在侧边栏配置部分添加，在现有API输入之后

st.sidebar.header("🌐 网络搜索配置")
st.session_state.use_web_search = st.sidebar.checkbox("启用网络搜索 fallback", value=st.session_state.use_web_search)

if st.session_state.use_web_search:
    exa_api_key = st.sidebar.text_input(
        "Exa AI API密钥",
        type="password",
        value=st.session_state.exa_api_key,
        help="当未找到相关文档时，网络搜索fallback需要此密钥"
    )
    st.session_state.exa_api_key = exa_api_key

    # 可选的域名过滤
    default_domains = ["arxiv.org", "wikipedia.org", "github.com", "medium.com"]
    custom_domains = st.sidebar.text_input(
        "自定义域名（逗号分隔）",
        value=",".join(default_domains),
        help="输入要搜索的域名，例如：arxiv.org,wikipedia.org"
    )
    search_domains = [d.strip() for d in custom_domains.split(",") if d.strip()]


# 搜索配置移至RAG模式检查内


# 工具函数
def init_qdrant() -> QdrantClient | None:
    """使用配置的设置初始化Qdrant客户端。

    返回:
        QdrantClient: 初始化成功的Qdrant客户端。
        None: 如果初始化失败。
    """
    if not all([st.session_state.qdrant_api_key, st.session_state.qdrant_url]):
        return None
    try:
        return QdrantClient(
            url=st.session_state.qdrant_url,
            api_key=st.session_state.qdrant_api_key,
            timeout=60
        )
    except Exception as e:
        st.error(f"🔴 Qdrant连接失败: {str(e)}")
        return None


# 文档处理函数
def process_pdf(file) -> List:
    """处理PDF文件并添加源元数据。"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file.getvalue())
            loader = PyPDFLoader(tmp_file.name)
            documents = loader.load()

            # 添加源元数据
            for doc in documents:
                doc.metadata.update({
                    "source_type": "pdf",
                    "file_name": file.name,
                    "timestamp": datetime.now().isoformat()
                })

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            return text_splitter.split_documents(documents)
    except Exception as e:
        st.error(f"📄 PDF处理错误: {str(e)}")
        return []


def process_web(url: str) -> List:
    """处理网页URL并添加源元数据。"""
    try:
        loader = WebBaseLoader(
            web_paths=(url,),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header", "content", "main")
                )
            )
        )
        documents = loader.load()

        # 添加源元数据
        for doc in documents:
            doc.metadata.update({
                "source_type": "url",
                "url": url,
                "timestamp": datetime.now().isoformat()
            })

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        return text_splitter.split_documents(documents)
    except Exception as e:
        st.error(f"🌐 网页处理错误: {str(e)}")
        return []


# 向量存储管理
def create_vector_store(client, texts):
    """创建并初始化带有文档的向量存储。"""
    try:
        # 必要时创建集合
        try:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=1024,
                    distance=Distance.COSINE
                )
            )
            st.success(f"📚 创建新集合: {COLLECTION_NAME}")
        except Exception as e:
            if "already exists" not in str(e).lower():
                raise e

        # 初始化向量存储
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=OllamaEmbedderr()
        )

        # 添加文档
        with st.spinner('📤 正在上传文档到Qdrant...'):
            vector_store.add_documents(texts)
            st.success("✅ 文档存储成功！")
            return vector_store

    except Exception as e:
        st.error(f"🔴 向量存储错误: {str(e)}")
        return None


def get_web_search_agent() -> Agent:
    """初始化网络搜索代理。"""
    return Agent(
        name="网络搜索代理",
        model=Ollama(id="llama3.2"),
        tools=[ExaTools(
            api_key=st.session_state.exa_api_key,
            include_domains=search_domains,
            num_results=5
        )],
        instructions="""你是网络搜索专家。你的任务是：
        1. 在网络上搜索与查询相关的信息
        2. 汇编并总结最相关的信息
        3. 在你的回答中包含来源
        """,
        show_tool_calls=True,
        markdown=True,
    )


def get_rag_agent() -> Agent:
    """初始化主RAG代理。"""
    return Agent(
        name="DeepSeek RAG代理",
        model=Ollama(id=st.session_state.model_version),
        instructions="""你是一个智能代理，专门提供准确的答案。

        当被问到问题时：
        - 分析问题并用你所知道的知识回答。

        当给出文档上下文时：
        - 专注于提供的文档中的信息
        - 要精确并引用具体细节

        当给出网络搜索结果时：
        - 清楚地表明信息来自网络搜索
        - 清晰地综合信息

        始终保持回答的高度准确性和清晰度。
        """,
        show_tool_calls=True,
        markdown=True,
    )


def check_document_relevance(query: str, vector_store, threshold: float = 0.7) -> tuple[bool, List]:
    if not vector_store:
        return False, []

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": threshold}
    )
    docs = retriever.invoke(query)
    return bool(docs), docs


chat_col, toggle_col = st.columns([0.9, 0.1])

with chat_col:
    prompt = st.chat_input("询问关于你的文档..." if st.session_state.rag_enabled else "问我任何事情...")

with toggle_col:
    st.session_state.force_web_search = st.toggle('🌐', help="强制网络搜索")

# 检查RAG是否启用
if st.session_state.rag_enabled:
    qdrant_client = init_qdrant()

    # 文件/URL上传部分
    st.sidebar.header("📁 数据上传")
    uploaded_file = st.sidebar.file_uploader("上传PDF", type=["pdf"])
    web_url = st.sidebar.text_input("或输入URL")

    # 处理文档
    if uploaded_file:
        file_name = uploaded_file.name
        if file_name not in st.session_state.processed_documents:
            with st.spinner('处理PDF中...'):
                texts = process_pdf(uploaded_file)
                if texts and qdrant_client:
                    if st.session_state.vector_store:
                        st.session_state.vector_store.add_documents(texts)
                    else:
                        st.session_state.vector_store = create_vector_store(qdrant_client, texts)
                    st.session_state.processed_documents.append(file_name)
                    st.success(f"✅ 添加PDF: {file_name}")

    if web_url:
        if web_url not in st.session_state.processed_documents:
            with st.spinner('处理URL中...'):
                texts = process_web(web_url)
                if texts and qdrant_client:
                    if st.session_state.vector_store:
                        st.session_state.vector_store.add_documents(texts)
                    else:
                        st.session_state.vector_store = create_vector_store(qdrant_client, texts)
                    st.session_state.processed_documents.append(web_url)
                    st.success(f"✅ 添加URL: {web_url}")

    # 在侧边栏显示来源
    if st.session_state.processed_documents:
        st.sidebar.header("📚 已处理来源")
        for source in st.session_state.processed_documents:
            if source.endswith('.pdf'):
                st.sidebar.text(f"📄 {source}")
            else:
                st.sidebar.text(f"🌐 {source}")

if prompt:
    # 将用户消息添加到历史记录
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    if st.session_state.rag_enabled:

        # 现有的RAG流程保持不变
        with st.spinner("🤔 正在评估查询..."):
            try:
                rewritten_query = prompt

                with st.expander("评估查询"):
                    st.write(f"用户提示: {prompt}")
            except Exception as e:
                st.error(f"❌ 重写查询错误: {str(e)}")
                rewritten_query = prompt

        # 步骤2: 基于force_web_search切换选择搜索策略
        context = ""
        docs = []
        if not st.session_state.force_web_search and st.session_state.vector_store:
            # 首先尝试文档搜索
            retriever = st.session_state.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": 5,
                    "score_threshold": st.session_state.similarity_threshold
                }
            )
            docs = retriever.invoke(rewritten_query)
            if docs:
                context = "\n\n".join([d.page_content for d in docs])
                st.info(f"📊 找到 {len(docs)} 个相关文档 (相似度 > {st.session_state.similarity_threshold})")
            elif st.session_state.use_web_search:
                st.info("🔄 在数据库中未找到相关文档， fallback到网络搜索...")

        # 步骤3: 在以下情况使用网络搜索:
        # 1. 通过切换强制启用网络搜索，或者
        # 2. 未找到相关文档且在设置中启用了网络搜索
        if (
                st.session_state.force_web_search or not context) and st.session_state.use_web_search and st.session_state.exa_api_key:
            with st.spinner("🔍 正在搜索网络..."):
                try:
                    web_search_agent = get_web_search_agent()
                    web_results = web_search_agent.run(rewritten_query).content
                    if web_results:
                        context = f"网络搜索结果:\n{web_results}"
                        if st.session_state.force_web_search:
                            st.info("ℹ️ 按切换请求使用网络搜索。")
                        else:
                            st.info("ℹ️ 由于未找到相关文档，使用网络搜索作为fallback。")
                except Exception as e:
                    st.error(f"❌ 网络搜索错误: {str(e)}")

        # 步骤4: 使用RAG代理生成响应
        with st.spinner("🤖 思考中..."):
            try:
                rag_agent = get_rag_agent()

                if context:
                    full_prompt = f"""上下文: {context}

原始问题: {prompt}
请基于可用信息提供全面的答案。"""
                else:
                    full_prompt = f"原始问题: {prompt}\n"
                    st.info("ℹ️ 在文档或网络搜索中未找到相关信息。")

                response = rag_agent.run(full_prompt)

                # 将助手响应添加到历史记录
                st.session_state.history.append({
                    "role": "assistant",
                    "content": response.content
                })

                # 显示助手响应
                with st.chat_message("assistant"):
                    st.write(response.content)

                    # 如果有来源则显示
                    if not st.session_state.force_web_search and 'docs' in locals() and docs:
                        with st.expander("🔍 查看文档来源"):
                            for i, doc in enumerate(docs, 1):
                                source_type = doc.metadata.get("source_type", "unknown")
                                source_icon = "📄" if source_type == "pdf" else "🌐"
                                source_name = doc.metadata.get("file_name" if source_type == "pdf" else "url",
                                                               "unknown")
                                st.write(f"{source_icon} 来源 {i} 来自 {source_name}:")
                                st.write(f"{doc.page_content[:200]}...")

            except Exception as e:
                st.error(f"❌ 生成响应错误: {str(e)}")

    else:
        # 无RAG的简单模式
        with st.spinner("🤖 思考中..."):
            try:
                rag_agent = get_rag_agent()
                web_search_agent = get_web_search_agent() if st.session_state.use_web_search else None

                # 如果强制或启用则处理网络搜索
                context = ""
                if st.session_state.force_web_search and web_search_agent:
                    with st.spinner("🔍 正在搜索网络..."):
                        try:
                            web_results = web_search_agent.run(prompt).content
                            if web_results:
                                context = f"网络搜索结果:\n{web_results}"
                                st.info("ℹ️ 按请求使用网络搜索。")
                        except Exception as e:
                            st.error(f"❌ 网络搜索错误: {str(e)}")

                # 生成响应
                if context:
                    full_prompt = f"""上下文: {context}

问题: {prompt}

请基于可用信息提供全面的答案。"""
                else:
                    full_prompt = prompt

                response = rag_agent.run(full_prompt)
                response_content = response.content

                # 提取思考过程和最终响应
                import re

                think_pattern = r'(.*?)<|FunctionCallEnd|>'
                think_match = re.search(think_pattern, response_content, re.DOTALL)

                if think_match:
                    thinking_process = think_match.group(1).strip()
                    final_response = re.sub(think_pattern, '', response_content, flags=re.DOTALL).strip()
                else:
                    thinking_process = None
                    final_response = response_content

                # 将助手响应添加到历史记录（仅最终响应）
                st.session_state.history.append({
                    "role": "assistant",
                    "content": final_response
                })

                # 显示助手响应
                with st.chat_message("assistant"):
                    if thinking_process:
                        with st.expander("🤔 查看思考过程"):
                            st.markdown(thinking_process)
                    st.markdown(final_response)

            except Exception as e:
                st.error(f"❌ 生成响应错误: {str(e)}")

else:
    st.warning("你可以直接与r1本地对话！切换RAG模式可上传文档！")