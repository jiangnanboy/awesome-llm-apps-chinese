import streamlit as st
from agno.agent import Agent, RunEvent
from agno.embedder.openai import OpenAIEmbedder
from agno.knowledge.url import UrlKnowledge
from agno.models.anthropic import Claude
from agno.tools.reasoning import ReasoningTools
from agno.vectordb.lancedb import LanceDb, SearchType
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

# 页面配置
st.set_page_config(
    page_title="带推理功能的智能体驱动型RAG系统",
    page_icon="🧐",
    layout="wide"
)

# 主标题和说明
st.title("🧐 带推理功能的智能体驱动型RAG系统")
st.markdown("""
本应用展示了一个AI智能体，它能够：
1. **检索**：从知识源中获取相关信息
2. **推理**：逐步分析信息
3. **回答**：附带引用地解答您的问题

在下方输入您的API密钥即可开始使用！
""")

# API密钥部分
st.subheader("🔑 API密钥")
col1, col2 = st.columns(2)
with col1:
    anthropic_key = st.text_input(
        "Anthropic API密钥",
        type="password",
        value=os.getenv("ANTHROPIC_API_KEY", ""),
        help="从https://console.anthropic.com/获取您的密钥"
    )
with col2:
    openai_key = st.text_input(
        "OpenAI API密钥",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="从https://platform.openai.com/获取您的密钥"
    )

# 检查是否提供了API密钥
if anthropic_key and openai_key:

    # 初始化知识库（缓存以避免重复加载）
    @st.cache_resource(show_spinner="📚 正在加载知识库...")
    def load_knowledge() -> UrlKnowledge:
        """加载并初始化带有向量数据库的知识库"""
        kb = UrlKnowledge(
            urls=["https://docs.agno.com/introduction/agents.md"],  # 默认URL
            vector_db=LanceDb(
                uri="tmp/lancedb",
                table_name="agno_docs",
                search_type=SearchType.vector,  # 使用向量搜索
                embedder=OpenAIEmbedder(
                    api_key=openai_key
                ),
            ),
        )
        kb.load(recreate=True)  # 将文档加载到向量数据库
        return kb


    # 初始化智能体（缓存以避免重复加载）
    @st.cache_resource(show_spinner="🤖 正在加载智能体...")
    def load_agent(_kb: UrlKnowledge) -> Agent:
        """创建具有推理能力的智能体"""
        return Agent(
            model=Claude(
                id="claude-sonnet-4-20250514",
                api_key=anthropic_key
            ),
            knowledge=_kb,
            search_knowledge=True,  # 启用知识搜索
            tools=[ReasoningTools(add_instructions=True)],  # 添加推理工具
            instructions=[
                "在回答中包含来源引用。",
                "回答问题前务必先搜索知识库。",
            ],
            markdown=True,  # 启用markdown格式
        )


    # 加载知识和智能体
    knowledge = load_knowledge()
    agent = load_agent(knowledge)

    # 侧边栏用于知识管理
    with st.sidebar:
        st.header("📚 知识来源")
        st.markdown("添加URL以扩展知识库：")

        # 显示当前URL
        st.write("**当前来源：**")
        for i, url in enumerate(knowledge.urls):
            st.text(f"{i + 1}. {url}")

        # 添加新URL
        st.divider()
        new_url = st.text_input(
            "添加新URL",
            placeholder="https://example.com/docs",
            help="输入要添加到知识库的URL"
        )

        if st.button("➕ 添加URL", type="primary"):
            if new_url:
                with st.spinner("📥 正在加载新文档..."):
                    knowledge.urls.append(new_url)
                    knowledge.load(
                        recreate=False,  # 不重建数据库
                        upsert=True,  # 更新现有文档
                        skip_existing=True  # 跳过已加载的文档
                    )
                st.success(f"✅ 已添加：{new_url}")
                st.rerun()  # 刷新以显示新URL
            else:
                st.error("请输入URL")

    # 主查询部分
    st.divider()
    st.subheader("🤔 提出问题")

    # 查询输入
    query = st.text_area(
        "您的问题：",
        value="什么是智能体（Agents）？",
        height=100,
        help="询问有关已加载知识来源的任何问题"
    )

    # 运行按钮
    if st.button("🚀 获取带推理过程的答案", type="primary"):
        if query:
            # 创建用于流式更新的容器
            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("### 🧠 推理过程")
                reasoning_container = st.container()
                reasoning_placeholder = reasoning_container.empty()

            with col2:
                st.markdown("### 💡 答案")
                answer_container = st.container()
                answer_placeholder = answer_container.empty()

            # 用于累积内容的变量
            citations = []
            answer_text = ""
            reasoning_text = ""

            # 流式传输智能体的响应
            with st.spinner("🔍 正在搜索和推理..."):
                for chunk in agent.run(
                        query,
                        stream=True,  # 启用流式传输
                        show_full_reasoning=True,  # 显示推理步骤
                        stream_intermediate_steps=True,  # 流式传输中间更新
                ):
                    # 更新推理过程显示
                    if chunk.reasoning_content:
                        reasoning_text = chunk.reasoning_content
                        reasoning_placeholder.markdown(
                            reasoning_text,
                            unsafe_allow_html=True
                        )

                    # 更新答案显示
                    if chunk.content and chunk.event in {RunEvent.run_response, RunEvent.run_completed}:
                        if isinstance(chunk.content, str):
                            answer_text += chunk.content
                            answer_placeholder.markdown(
                                answer_text,
                                unsafe_allow_html=True
                            )

                    # 收集引用
                    if chunk.citations and chunk.citations.urls:
                        citations = chunk.citations.urls

            # 如果有引用，显示引用
            if citations:
                st.divider()
                st.subheader("📚 来源")
                for cite in citations:
                    title = cite.title or cite.url
                    st.markdown(f"- [{title}]({cite.url})")
        else:
            st.error("请输入问题")

else:
    # 如果缺少API密钥，显示说明
    st.info("""
    👋 **欢迎！要使用此应用，您需要：**

    1. **Anthropic API密钥** - 用于Claude AI模型
       - 在[console.anthropic.com](https://console.anthropic.com/)注册

    2. **OpenAI API密钥** - 用于嵌入生成
       - 在[platform.openai.com](https://platform.openai.com/)注册

    获得这两个密钥后，在上方输入即可开始使用！
    """)

# 带解释的页脚
st.divider()
with st.expander("📖 工作原理"):
    st.markdown("""
    **本应用使用Agno框架创建智能问答系统：**

    1. **知识加载**：URL被处理并存储在向量数据库（LanceDB）中
    2. **向量搜索**：使用OpenAI的嵌入进行语义搜索，以找到相关信息
    3. **推理工具**：智能体使用特殊工具逐步思考问题
    4. **Claude AI**：Anthropic的Claude模型处理信息并生成答案

    **核心组件：**
    - `UrlKnowledge`：管理从URL加载文档
    - `LanceDb`：用于高效相似性搜索的向量数据库
    - `OpenAIEmbedder`：使用OpenAI的嵌入模型将文本转换为嵌入
    - `ReasoningTools`：支持逐步推理
    - `Agent`：协调所有工作以回答问题
    """)