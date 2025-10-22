import streamlit as st
import os
from agno.agent import Agent
from agno.embedder.openai import OpenAIEmbedder
from agno.knowledge.url import UrlKnowledge
from agno.models.openai import OpenAIChat
from agno.vectordb.lancedb import LanceDb, SearchType

from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 页面配置
st.set_page_config(
    page_title="基于GPT-5的智能体驱动型RAG系统",
    page_icon="🧠",
    layout="wide"
)

# 主标题和描述
st.title("🧠 基于GPT-5的智能体驱动型RAG系统")
st.markdown("""
本应用展示了一个智能AI代理，它能够：
1. **检索**：使用LanceDB从知识源中获取相关信息
2. **回答**：清晰简洁地解答您的问题

在侧边栏输入您的OpenAI API密钥即可开始使用！
""")

# 侧边栏（用于API密钥和设置）
with st.sidebar:
    st.header("🔧 配置")

    # OpenAI API密钥
    openai_key = st.text_input(
        "OpenAI API密钥",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="从https://platform.openai.com/获取您的密钥"
    )

    # 向知识库添加URL
    st.subheader("🌐 添加知识来源")
    new_url = st.text_input(
        "添加URL",
        placeholder="https://docs.agno.com/introduction",
        help="输入要添加到知识库的URL"
    )

    if st.button("➕ 添加URL", type="primary"):
        if new_url:
            st.session_state.urls_to_add = new_url
            st.success(f"URL已添加到队列：{new_url}")
        else:
            st.error("请输入URL")

# 检查是否提供了API密钥
if openai_key:
    # 初始化知识库（缓存以避免重复加载）
    @st.cache_resource(show_spinner="📚 正在加载知识库...")
    def load_knowledge() -> UrlKnowledge:
        """加载并初始化带有LanceDB的知识库"""
        kb = UrlKnowledge(
            urls=["https://docs.agno.com/introduction/agents.md"],  # 默认URL
            vector_db=LanceDb(
                uri="tmp/lancedb",
                table_name="agentic_rag_docs",
                search_type=SearchType.vector,  # 使用向量搜索
                embedder=OpenAIEmbedder(
                    api_key=openai_key
                ),
            ),
        )
        kb.load(recreate=True)  # 将文档加载到LanceDB
        return kb


    # 初始化智能体（缓存以避免重复加载）
    @st.cache_resource(show_spinner="🤖 正在加载智能体...")
    def load_agent(_kb: UrlKnowledge) -> Agent:
        """创建具有推理能力的智能体"""
        return Agent(
            model=OpenAIChat(
                id="gpt-5-nano",
                api_key=openai_key
            ),
            knowledge=_kb,
            search_knowledge=True,  # 启用知识搜索
            instructions=[
                "回答问题前务必先搜索您的知识库。",
                "以markdown格式提供清晰、结构良好的答案。",
                "在适当的地方使用适当的markdown格式，包括标题、列表和强调。",
                "在有帮助时，使用清晰的部分和项目符号组织您的回应。",
            ],
            markdown=True,  # 启用markdown格式
        )


    # 加载知识和智能体
    knowledge = load_knowledge()
    agent = load_agent(knowledge)

    # 显示知识库中当前的URL
    if knowledge.urls:
        st.sidebar.subheader("📚 当前知识来源")
        for i, url in enumerate(knowledge.urls, 1):
            st.sidebar.markdown(f"{i}. {url}")

    # 处理URL添加
    if hasattr(st.session_state, 'urls_to_add') and st.session_state.urls_to_add:
        with st.spinner("📥 正在加载新文档..."):
            knowledge.urls.append(st.session_state.urls_to_add)
            knowledge.load(
                recreate=False,  # 不重建数据库
                upsert=True,  # 更新现有文档
                skip_existing=True  # 跳过已加载的文档
            )
        st.success(f"✅ 已添加：{st.session_state.urls_to_add}")
        del st.session_state.urls_to_add
        st.rerun()

    # 主查询部分
    st.divider()
    st.subheader("🤔 提出问题")

    # 建议的提示
    st.markdown("**尝试这些提示：**")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("什么是Agno？", use_container_width=True):
            st.session_state.query = "什么是Agno？智能体是如何工作的？"
    with col2:
        if st.button("Agno中的团队", use_container_width=True):
            st.session_state.query = "Agno中的团队是什么？它们是如何工作的？"
    with col3:
        if st.button("构建RAG系统", use_container_width=True):
            st.session_state.query = "给我一个构建RAG系统的分步指南。"

    # 查询输入
    query = st.text_area(
        "您的问题：",
        value=st.session_state.get("query", "什么是AI智能体？"),
        height=100,
        help="询问有关已加载知识来源的任何问题"
    )

    # 运行按钮
    if st.button("🚀 获取答案", type="primary"):
        if query:
            # 创建答案容器
            st.markdown("### 💡 答案")
            answer_container = st.container()
            answer_placeholder = answer_container.empty()

            # 用于累积内容的变量
            answer_text = ""

            # 流式传输智能体的响应
            with st.spinner("🔍 正在搜索并生成答案..."):
                for chunk in agent.run(
                        query,
                        stream=True,  # 启用流式传输
                ):
                    # 更新答案显示 - 仅显示来自RunResponseContent事件的内容
                    if hasattr(chunk, 'event') and chunk.event == "RunResponseContent":
                        if hasattr(chunk, 'content') and chunk.content and isinstance(chunk.content, str):
                            answer_text += chunk.content
                            answer_placeholder.markdown(
                                answer_text,
                                unsafe_allow_html=True
                            )
        else:
            st.error("请输入问题")

else:
    # 如果缺少API密钥，显示说明
    st.info("""
    👋 **欢迎！要使用此应用，您需要：**

    - **OpenAI API密钥**（在侧边栏中设置）
      - 在[platform.openai.com](https://platform.openai.com/)注册
      - 生成新的API密钥

    输入密钥后，应用将加载知识库和智能体。
    """)

# 带解释的页脚
st.divider()
with st.expander("📖 工作原理"):
    st.markdown("""
    **本应用使用Agno框架创建智能问答系统：**

    1. **知识加载**：URL被处理并存储在LanceDB向量数据库中
    2. **向量搜索**：使用OpenAI的嵌入进行语义搜索，以找到相关信息
    3. **GPT-5**：OpenAI的GPT-5模型处理信息并生成答案

    **核心组件：**
    - `UrlKnowledge`：管理从URL加载文档
    - `LanceDb`：用于高效相似性搜索的向量数据库
    - `OpenAIEmbedder`：使用OpenAI的嵌入模型将文本转换为嵌入

    - `Agent`：协调所有工作以回答问题

    **为什么选择LanceDB？**
    - 轻量级且易于设置
    - 不需要外部数据库
    - 快速的向量搜索能力
    - 非常适合原型设计和中小型应用
    """)