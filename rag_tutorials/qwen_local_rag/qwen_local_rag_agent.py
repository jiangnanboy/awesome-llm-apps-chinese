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
            model_name (str): 用于生成嵌入向量的模型名称。
        """
        self.embedder = OllamaEmbedder(id=model_name, dimensions=1024)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self.embedder.get_embedding(text)


# 常量定义
COLLECTION_NAME = "test-qwen-r1"

# Streamlit应用初始化
st.title("🐋 Qwen 3 本地RAG推理助手")

# --- 添加模型信息框 ---
st.info("**Qwen3**：通义千问系列最新一代大语言模型，提供完整的稠密模型和混合专家（MoE）模型套件。")
st.info("**Gemma 3**：多模态模型（支持文本和图像处理），拥有128K上下文窗口，支持超过140种语言。")
# -------------------------

# 会话状态初始化
if 'model_version' not in st.session_state:
    st.session_state.model_version = "qwen3:1.7b"  # 默认使用轻量级模型
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
    st.session_state.rag_enabled = True  # 默认启用RAG功能

# 侧边栏配置
st.sidebar.header("⚙️ 设置")

# 模型选择
st.sidebar.header("🧠 模型选择")
model_help = """
- qwen3:1.7b: 轻量级模型（MoE架构）
- gemma3:1b: 性能更强，但需要更好的GPU/内存（32K上下文窗口）
- gemma3:4b: 性能更强且支持多模态（视觉）（128K上下文窗口）
- deepseek-r1:1.5b: 深度求索R1模型（1.5B参数）
- qwen3:8b: 性能最强，但需要高性能GPU/内存

根据您的硬件性能选择合适的模型。
"""
st.session_state.model_version = st.sidebar.radio(
    "选择模型版本",
    options=["qwen3:1.7b", "gemma3:1b", "gemma3:4b", "deepseek-r1:1.5b", "qwen3:8b"],
    help=model_help
)

st.sidebar.info("请先执行命令拉取模型：ollama pull qwen3:1.7b")

# RAG模式切换
st.sidebar.header("📚 RAG模式")
st.session_state.rag_enabled = st.sidebar.toggle("启用RAG", value=st.session_state.rag_enabled)

# 清空聊天按钮
if st.sidebar.button("✨ 清空聊天记录"):
    st.session_state.history = []
    st.rerun()

# 仅当RAG启用时显示API配置
if st.session_state.rag_enabled:
    st.sidebar.header("🔬 搜索调优")
    st.session_state.similarity_threshold = st.sidebar.slider(
        "相似度阈值",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        help="值越低返回的文档越多但相关性可能越低；值越高相关性要求越严格。"
    )

# 在侧边栏配置区添加网页搜索设置（现有API输入之后）
st.sidebar.header("🌍 网页搜索")
st.session_state.use_web_search = st.sidebar.checkbox("启用网页搜索 fallback", value=st.session_state.use_web_search)

if st.session_state.use_web_search:
    exa_api_key = st.sidebar.text_input(
        "Exa AI API密钥",
        type="password",
        value=st.session_state.exa_api_key,
        help="当未找到相关文档时，需要此密钥启用网页搜索 fallback 功能"
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


# 工具函数
def init_qdrant() -> QdrantClient | None:
    """初始化本地Docker部署的Qdrant客户端。

    返回:
        QdrantClient: 初始化成功的Qdrant客户端。
        None: 初始化失败时返回None。
    """
    try:
        return QdrantClient(url="http://localhost:6333")
    except Exception as e:
        st.error(f"🔴 Qdrant连接失败: {str(e)}")
        return None


# 文档处理函数
def process_pdf(file) -> List:
    """处理PDF文件并添加来源元数据。"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file.getvalue())
            loader = PyPDFLoader(tmp_file.name)
            documents = loader.load()

            # 添加来源元数据
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
    """处理网页URL并添加来源元数据。"""
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

        # 添加来源元数据
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
    """创建并初始化包含文档的向量存储。"""
    try:
        # 按需创建集合
        try:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=1024,  # 与嵌入模型维度匹配
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
        with st.spinner('📤 正在将文档上传到Qdrant...'):
            vector_store.add_documents(texts)
            st.success("✅ 文档存储成功！")
            return vector_store

    except Exception as e:
        st.error(f"🔴 向量存储错误: {str(e)}")
        return None


def get_web_search_agent() -> Agent:
    """初始化网页搜索代理。"""
    return Agent(
        name="网页搜索代理",
        model=Ollama(id="llama3.2"),
        tools=[ExaTools(
            api_key=st.session_state.exa_api_key,
            include_domains=search_domains,
            num_results=5
        )],
        instructions="""你是网页搜索专家，任务如下：
        1. 根据查询关键词搜索相关网页信息
        2. 整理并总结最相关的信息内容
        3. 在回复中包含信息来源链接
        """,
        show_tool_calls=True,
        markdown=True,
    )


def get_rag_agent() -> Agent:
    """初始化主RAG代理。"""
    return Agent(
        name="Qwen 3 RAG代理",
        model=Ollama(id=st.session_state.model_version),
        instructions="""你是专注于提供准确答案的智能代理。

        当收到问题时：
        - 分析问题并利用已有知识回答
        - 若提供文档上下文，优先基于文档内容回答
        - 回答需精确并引用具体细节

        当收到网页搜索结果时：
        - 明确标注信息来源于网页搜索
        - 清晰整合搜索到的信息

        始终保持回答的高准确性和清晰度。
        """,
        show_tool_calls=True,
        markdown=True,
    )


def check_document_relevance(query: str, vector_store, threshold: float = 0.7) -> tuple[bool, List]:
    """检查查询与文档的相关性（原代码中定义但未使用，保留用于后续扩展）。

    参数:
        query (str): 用户查询
        vector_store: 向量存储实例
        threshold (float): 相关性阈值

    返回:
        tuple[bool, List]: 相关性结果（是否相关，相关文档列表）
    """
    if not vector_store:
        return False, []

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": threshold}
    )
    docs = retriever.invoke(query)
    return bool(docs), docs


# 布局：聊天输入框与网页搜索强制切换按钮
chat_col, toggle_col = st.columns([0.9, 0.1])

with chat_col:
    prompt = st.chat_input("请提问（已启用RAG，可询问文档相关问题）" if st.session_state.rag_enabled else "请提问...")

with toggle_col:
    st.session_state.force_web_search = st.toggle('🌐', help="强制使用网页搜索")

# 检查RAG是否启用
if st.session_state.rag_enabled:
    qdrant_client = init_qdrant()

    # --- 文档上传区域（移至主内容区）---
    with st.expander("📁 上传文档或URL用于RAG", expanded=False):
        if not qdrant_client:
            st.warning("⚠️ 请在侧边栏配置Qdrant API密钥和URL，以启用文档处理功能。")
        else:
            uploaded_files = st.file_uploader(
                "上传PDF文件",
                accept_multiple_files=True,
                type='pdf'
            )
            url_input = st.text_input("输入网页URL以抓取内容")

            if uploaded_files:
                st.write(f"正在处理 {len(uploaded_files)} 个PDF文件...")
                all_texts = []
                for file in uploaded_files:
                    if file.name not in st.session_state.processed_documents:
                        with st.spinner(f"正在处理 {file.name}... "):
                            texts = process_pdf(file)
                            if texts:
                                all_texts.extend(texts)
                                st.session_state.processed_documents.append(file.name)
                    else:
                        st.write(f"📄 {file.name} 已处理过，跳过。")

                if all_texts:
                    with st.spinner("正在创建向量存储..."):
                        st.session_state.vector_store = create_vector_store(qdrant_client, all_texts)

            if url_input:
                if url_input not in st.session_state.processed_documents:
                    with st.spinner(f"正在抓取并处理 {url_input}..."):
                        texts = process_web(url_input)
                        if texts:
                            st.session_state.vector_store = create_vector_store(qdrant_client, texts)
                            st.session_state.processed_documents.append(url_input)
                else:
                    st.write(f"🔗 {url_input} 已处理过，跳过。")

            if st.session_state.vector_store:
                st.success("向量存储已就绪，可以开始提问。")
            elif not uploaded_files and not url_input:
                st.info("请上传PDF文件或输入URL，以构建向量存储。")

    # 在侧边栏显示已处理的来源
    if st.session_state.processed_documents:
        st.sidebar.header("📚 已处理来源")
        for source in st.session_state.processed_documents:
            if source.endswith('.pdf'):
                st.sidebar.text(f"📄 {source}")
            else:
                st.sidebar.text(f"🌐 {source}")

# 处理用户输入
if prompt:
    # 将用户消息添加到历史记录
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    if st.session_state.rag_enabled:
        # 现有RAG流程保持不变
        with st.spinner("🤔 正在分析查询..."):
            try:
                rewritten_query = prompt  # 此处可扩展查询重写逻辑
                with st.expander("查询分析详情"):
                    st.write(f"用户原始查询: {prompt}")
            except Exception as e:
                st.error(f"❌ 查询重写错误: {str(e)}")
                rewritten_query = prompt

        # 步骤2：根据强制网页搜索切换按钮选择搜索策略
        context = ""
        docs = []
        if not st.session_state.force_web_search and st.session_state.vector_store:
            # 优先尝试文档搜索
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
                st.info(f"📊 找到 {len(docs)} 个相关文档（相似度 > {st.session_state.similarity_threshold}）")
            elif st.session_state.use_web_search:
                st.info("🔄 数据库中未找到相关文档，正在切换到网页搜索...")

        # 步骤3：满足以下条件时使用网页搜索：
        # 1. 强制网页搜索按钮已开启，或
        # 2. 未找到相关文档且设置中已启用网页搜索
        if (
                st.session_state.force_web_search or not context) and st.session_state.use_web_search and st.session_state.exa_api_key:
            with st.spinner("🔍 正在网页搜索..."):
                try:
                    web_search_agent = get_web_search_agent()
                    web_results = web_search_agent.run(rewritten_query).content
                    if web_results:
                        context = f"网页搜索结果:\n{web_results}"
                        if st.session_state.force_web_search:
                            st.info("ℹ️ 已按请求使用网页搜索。")
                        else:
                            st.info("ℹ️ 因未找到相关文档，已使用网页搜索 fallback。")
                except Exception as e:
                    st.error(f"❌ 网页搜索错误: {str(e)}")

        # 步骤4：使用RAG代理生成回答
        with st.spinner("🤖 正在生成回答..."):
            try:
                rag_agent = get_rag_agent()

                if context:
                    full_prompt = f"""上下文: {context}

原始问题: {prompt}
请基于可用信息提供全面的回答。"""
                else:
                    full_prompt = f"原始问题: {prompt}\n"
                    st.info("ℹ️ 在文档和网页搜索中均未找到相关信息。")

                response = rag_agent.run(full_prompt)

                # 将助手回答添加到历史记录
                st.session_state.history.append({
                    "role": "assistant",
                    "content": response.content
                })

                # 显示助手回答
                with st.chat_message("assistant"):
                    st.write(response.content)

                    # 若存在相关文档，显示来源（未强制网页搜索时）
                    if not st.session_state.force_web_search and 'docs' in locals() and docs:
                        with st.expander("🔍 查看文档来源"):
                            for i, doc in enumerate(docs, 1):
                                source_type = doc.metadata.get("source_type", "unknown")
                                source_icon = "📄" if source_type == "pdf" else "🌐"
                                source_name = doc.metadata.get("file_name" if source_type == "pdf" else "url",
                                                               "unknown")
                                st.write(f"{source_icon} 来源 {i}（{source_name}）:")
                                st.write(f"{doc.page_content[:200]}...")

            except Exception as e:
                st.error(f"❌ 生成回答错误: {str(e)}")

    else:
        # 无RAG的简单模式
        with st.spinner("🤖 正在生成回答..."):
            try:
                rag_agent = get_rag_agent()
                web_search_agent = get_web_search_agent() if st.session_state.use_web_search else None

                # 若强制或启用网页搜索，先执行网页搜索
                context = ""
                if st.session_state.force_web_search and web_search_agent:
                    with st.spinner("🔍 正在网页搜索..."):
                        try:
                            web_results = web_search_agent.run(prompt).content
                            if web_results:
                                context = f"网页搜索结果:\n{web_results}"
                                st.info("ℹ️ 已按请求使用网页搜索。")
                        except Exception as e:
                            st.error(f"❌ 网页搜索错误: {str(e)}")

                # 生成回答
                if context:
                    full_prompt = f"""上下文: {context}

问题: {prompt}

请基于可用信息提供全面的回答。"""
                else:
                    full_prompt = prompt

                response = rag_agent.run(full_prompt)
                response_content = response.content

                # 提取思考过程和最终回答（使用正则表达式）
                import re

                think_pattern = r'(.*?)'
                think_match = re.search(think_pattern, response_content, re.DOTALL)

                if think_match:
                    thinking_process = think_match.group(1).strip()
                    final_response = re.sub(think_pattern, '', response_content, flags=re.DOTALL).strip()
                else:
                    thinking_process = None
                    final_response = response_content

                # 将助手的最终回答添加到历史记录
                st.session_state.history.append({
                    "role": "assistant",
                    "content": final_response
                })

                # 显示助手回答
                with st.chat_message("assistant"):
                    if thinking_process:
                        with st.expander("🤔 查看思考过程"):
                            st.markdown(thinking_process)
                    st.markdown(final_response)

            except Exception as e:
                st.error(f"❌ 生成回答错误: {str(e)}")

else:
    st.warning("您可以直接与本地的qwen和gemma模型对话！切换RAG模式可上传文档进行问答！")