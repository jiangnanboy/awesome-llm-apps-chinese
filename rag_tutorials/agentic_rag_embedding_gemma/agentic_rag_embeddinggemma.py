import streamlit as st
from agno.agent import Agent  # 导入Agno框架的智能体类
from agno.embedder.ollama import OllamaEmbedder  # 导入基于Ollama的嵌入生成器
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase  # 导入从PDF链接加载知识库的类
from agno.models.ollama import Ollama  # 导入基于Ollama的模型接口
from agno.vectordb.lancedb import LanceDb, SearchType  # 导入LanceDB向量数据库及搜索类型

# 页面配置
st.set_page_config(
    page_title="基于Google EmbeddingGemma的智能体驱动型RAG系统",  # 页面标题
    page_icon="🔥",  # 页面图标
    layout="wide"  # 宽屏布局
)


# 缓存知识库资源（避免重复加载）
@st.cache_resource
def load_knowledge_base(urls):
    # 初始化PDF链接知识库
    knowledge_base = PDFUrlKnowledgeBase(
        urls=urls,  # 传入PDF文件的URL列表
        vector_db=LanceDb(  # 配置LanceDB向量数据库
            table_name="recipes",  # 数据库表名（此处为示例名"食谱"，可根据需求修改）
            uri="tmp/lancedb",  # 数据库存储路径（本地临时目录）
            search_type=SearchType.vector,  # 搜索类型：向量搜索
            embedder=OllamaEmbedder(  # 配置嵌入生成器（使用EmbeddingGemma模型）
                id="embeddinggemma:latest",  # 模型ID（最新版EmbeddingGemma）
                dimensions=768,  # 嵌入向量维度（768维）
            ),
        ),
    )
    knowledge_base.load()  # 加载知识库（处理PDF并写入向量数据库）
    return knowledge_base


# 初始化会话状态中的URL列表（用于存储用户添加的PDF链接）
if 'urls' not in st.session_state:
    st.session_state.urls = []

# 加载知识库（从会话状态的URL列表中获取数据源）
kb = load_knowledge_base(st.session_state.urls)

# 初始化智能体（协调模型、知识库与交互逻辑）
agent = Agent(
    model=Ollama(id="llama3.2:latest"),  # 配置生成式模型（最新版Llama 3.2）
    knowledge=kb,  # 关联知识库（用于检索相关信息）
    instructions=[  # 智能体指令（定义回答规则）
        "从知识库中搜索相关信息，并基于这些信息生成回答。",
        "回答需清晰易懂，结构规整。",
        "适当使用标题、项目符号或编号列表优化可读性（如适用）。",
    ],
    search_knowledge=True,  # 启用知识库搜索（必须基于检索结果回答）
    show_tool_calls=False,  # 不显示工具调用细节（简化用户界面）
    markdown=True,  # 支持Markdown格式输出（优化排版）
)

# 侧边栏（用于添加知识库来源）
with st.sidebar:
    # 显示框架/工具图标（Google、Ollama、Agno）
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("google.png")  # Google（EmbeddingGemma模型所属）
    with col2:
        st.image("ollama.png")  # Ollama（本地模型运行工具）
    with col3:
        st.image("agno.png")  # Agno（智能体与RAG框架）

    st.header("🌐 添加知识库来源")
    # 输入PDF链接的文本框
    new_url = st.text_input(
        "添加URL",
        placeholder="https://example.com/sample.pdf",  # 示例链接
        help="输入PDF文件的URL以添加到知识库",  # 帮助提示
    )
    # "添加URL"按钮（触发知识库更新）
    if st.button("➕ 添加URL", type="primary"):
        if new_url:  # 若输入了有效URL
            kb.urls.append(new_url)  # 将URL添加到知识库的URL列表
            with st.spinner("📥 正在添加新URL..."):
                # 加载新URL对应的PDF（不重建数据库，仅追加数据）
                kb.load(recreate=False, upsert=True)
            st.success(f"✅ 已添加：{new_url}")  # 显示成功提示
        else:
            st.error("请输入有效的URL")  # 输入为空时显示错误提示

    # 显示当前已添加的知识库来源
    if kb.urls:
        st.subheader("📚 当前知识库来源")
        # 遍历URL列表并编号显示
        for i, url in enumerate(kb.urls, 1):
            st.markdown(f"{i}. {url}")

# 主页面标题与说明
st.title("🔥 基于EmbeddingGemma的智能体驱动型RAG系统（100%本地部署）")
st.markdown(
    """
本应用展示了基于Ollama本地模型的智能体驱动型RAG（检索增强生成）系统：

- **EmbeddingGemma**：用于生成文本嵌入向量（支撑语义搜索）
- **LanceDB**：本地向量数据库（存储嵌入向量与PDF文本片段）

在侧边栏添加PDF文件的URL，即可开始提问并获取基于PDF内容的回答。
    """
)

# 问题输入框（用户提问）
query = st.text_input("请输入您的问题：")

# 回答生成逻辑
if st.button("🚀 获取答案", type="primary"):
    if not query:  # 若未输入问题
        st.error("请输入您的问题")
    else:
        st.markdown("### 💡 回答")

        # 显示加载中状态
        with st.spinner("🔍 正在搜索知识库并生成回答..."):
            try:
                response = ""  # 存储完整回答
                resp_container = st.empty()  # 用于动态更新回答的容器
                # 流式获取智能体的回答（逐段显示，提升用户体验）
                gen = agent.run(query, stream=True)
                for resp_chunk in gen:
                    # 若当前片段有内容，则追加到完整回答并更新显示
                    if resp_chunk.content is not None:
                        response += resp_chunk.content
                        resp_container.markdown(response)
            except Exception as e:
                # 捕获异常并显示错误信息
                st.error(f"错误：{e}")

# "工作原理"展开面板（解释系统核心逻辑）
with st.expander("📖 系统工作原理"):
    st.markdown(
        """
**本应用基于Agno框架构建，核心是一个智能问答系统，工作流程如下：**

1. **知识库加载**：系统会处理用户添加的PDF URL，提取文本内容并生成嵌入向量，最终存储到LanceDB向量数据库中。
2. **EmbeddingGemma嵌入生成**：EmbeddingGemma模型负责将文本（PDF片段、用户问题）转换为高维向量，为语义搜索提供基础。
3. **Llama 3.2回答生成**：当用户提问时，系统先通过向量搜索从知识库中找到最相关的文本片段，再由Llama 3.2模型基于这些片段生成准确回答。

**核心组件说明：**
- `EmbeddingGemma`：嵌入模型，负责将文本转换为向量（支撑语义匹配）。
- `LanceDB`：轻量级本地向量数据库，高效存储和检索嵌入向量。
- `PDFUrlKnowledgeBase`：知识库管理组件，专门处理从PDF URL加载数据的逻辑。
- `OllamaEmbedder`：基于Ollama的嵌入生成工具，对接EmbeddingGemma模型。
- `Agno Agent`：智能体核心，协调"检索知识库→调用生成模型→组织回答"的全流程。
        """
    )