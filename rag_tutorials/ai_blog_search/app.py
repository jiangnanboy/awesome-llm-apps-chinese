from langchain_google_genai import GoogleGenerativeAIEmbeddings  # 导入谷歌生成式AI嵌入模型
from langchain_qdrant import QdrantVectorStore  # 导入LangChain的Qdrant向量存储模块
from qdrant_client import QdrantClient  # 导入Qdrant客户端
from uuid import uuid4  # 导入UUID生成工具（用于生成唯一文档ID）
from langchain_community.document_loaders import WebBaseLoader  # 导入网页文档加载器
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 导入递归字符文本分割器
from langchain.tools.retriever import create_retriever_tool  # 导入检索工具创建函数

from typing import Annotated, Literal, Sequence  # 导入类型注解相关工具
from typing_extensions import TypedDict  # 导入类型字典（用于定义状态结构）
from functools import partial  # 导入偏函数（用于固定函数参数）

from langchain import hub  # 导入LangChain中心库（用于拉取提示模板）
from langchain_core.messages import BaseMessage, HumanMessage  # 导入核心消息类型
from langgraph.graph.message import add_messages  # 导入LangGraph消息处理函数
from langchain_core.output_parsers import StrOutputParser  # 导入字符串输出解析器
from langchain_core.prompts import PromptTemplate  # 导入提示模板类
from langchain_google_genai import ChatGoogleGenerativeAI  # 导入谷歌生成式AI聊天模型

from pydantic import BaseModel, Field  # 导入Pydantic（用于数据模型验证）

from langgraph.graph import END, StateGraph, START  # 导入LangGraph核心组件（状态图、起始/结束节点）
from langgraph.prebuilt import ToolNode, tools_condition  # 导入LangGraph预置工具节点和条件判断

import streamlit as st  # 导入Streamlit（用于构建Web界面）

# 设置Streamlit页面配置
st.set_page_config(page_title="AI博客搜索", page_icon=":mag_right:")  # 页面标题和图标
st.header(":blue[基于LangGraph的智能检索增强生成（Agentic RAG）：] :green[AI博客搜索]")  # 页面头部标题

# 初始化会话状态变量（若不存在则创建）
if 'qdrant_host' not in st.session_state:  # Qdrant服务地址
    st.session_state.qdrant_host = ""
if 'qdrant_api_key' not in st.session_state:  # Qdrant API密钥
    st.session_state.qdrant_api_key = ""
if 'gemini_api_key' not in st.session_state:  # Gemini API密钥
    st.session_state.gemini_api_key = ""


def set_sidebar():
    """设置侧边栏（用于API密钥和配置输入）"""
    with st.sidebar:  # 在侧边栏中创建内容
        st.subheader("API配置")  # 侧边栏子标题

        # 输入Qdrant服务地址（密码类型输入框，隐藏输入内容）
        qdrant_host = st.text_input("输入您的Qdrant服务地址（Host URL）：", type="password")
        # 输入Qdrant API密钥
        qdrant_api_key = st.text_input("输入您的Qdrant API密钥：", type="password")
        # 输入Gemini API密钥
        gemini_api_key = st.text_input("输入您的Gemini API密钥：", type="password")

        # "完成"按钮（点击后保存配置到会话状态）
        if st.button("完成"):
            # 验证所有输入框均已填写
            if qdrant_host and qdrant_api_key and gemini_api_key:
                # 将输入值保存到会话状态（跨页面/刷新后仍保留）
                st.session_state.qdrant_host = qdrant_host
                st.session_state.qdrant_api_key = qdrant_api_key
                st.session_state.gemini_api_key = gemini_api_key
                st.success("API密钥保存成功！")  # 显示成功提示
            else:
                st.warning("请填写所有API配置项")  # 显示警告提示


def initialize_components():
    """初始化需要API密钥的组件（嵌入模型、Qdrant客户端、向量存储）"""
    # 检查会话状态中是否已填写所有必要的API配置
    if not all([st.session_state.qdrant_host,
                st.session_state.qdrant_api_key,
                st.session_state.gemini_api_key]):
        return None, None, None  # 未填写则返回空值

    try:
        # 初始化谷歌生成式AI嵌入模型（使用Gemini API密钥）
        embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",  # 指定嵌入模型版本
            google_api_key=st.session_state.gemini_api_key  # 传入API密钥
        )

        # 初始化Qdrant客户端（连接Qdrant服务）
        client = QdrantClient(
            st.session_state.qdrant_host,  # Qdrant服务地址
            api_key=st.session_state.qdrant_api_key  # Qdrant API密钥
        )

        # 初始化Qdrant向量存储（关联客户端、集合名和嵌入模型）
        db = QdrantVectorStore(
            client=client,  # 已初始化的Qdrant客户端
            collection_name="qdrant_db",  # 向量集合名称（存储文档向量的表）
            embedding=embedding_model  # 用于生成文档向量的嵌入模型
        )

        return embedding_model, client, db  # 返回初始化成功的组件

    except Exception as e:
        st.error(f"初始化失败：{str(e)}")  # 捕获异常并显示错误信息
        return None, None, None  # 初始化失败则返回空值


class AgentState(TypedDict):
    """定义智能体（Agent）的状态结构（基于TypedDict实现类型约束）"""
    # 消息序列：使用add_messages注解确保消息可追加（不覆盖历史）
    messages: Annotated[Sequence[BaseMessage], add_messages]


# 定义LangGraph的边（Edge）：用于节点间的条件跳转逻辑
## 文档相关性判断边
def grade_documents(state) -> Literal["generate", "rewrite"]:
    """
    判断检索到的文档是否与用户问题相关，决定后续流程走向。

    参数：
        state (dict): 当前智能体状态（包含消息历史）

    返回：
        str: 流程决策结果（"generate"表示文档相关，直接生成回答；"rewrite"表示文档不相关，需重写问题）
    """

    print("---开始文档相关性判断---")

    # 定义文档相关性评分的数据模型（使用Pydantic确保输出格式正确）
    class grade(BaseModel):
        """文档相关性的二元评分模型"""
        binary_score: str = Field(description="相关性评分，仅允许'yes'（相关）或'no'（不相关）")

    # 初始化谷歌生成式AI聊天模型（用于判断相关性，低温确保结果稳定）
    model = ChatGoogleGenerativeAI(
        api_key=st.session_state.gemini_api_key,  # 传入API密钥
        temperature=0,  # 温度参数（0表示确定性输出，无随机性）
        model="gemini-2.0-flash",  # 指定Gemini模型版本（轻量版，响应快）
        streaming=True  # 启用流式输出（逐步返回结果）
    )

    # 为模型绑定结构化输出（强制模型按grade类的格式返回结果）
    llm_with_tool = model.with_structured_output(grade)

    # 定义相关性判断的提示模板（明确模型的判断规则）
    prompt = PromptTemplate(
        template="""你是一名文档相关性评估员，负责判断检索到的文档是否与用户问题相关。
        以下是检索到的文档内容：
        \n\n {context} \n\n
        以下是用户的问题：{question} \n
        判断规则：若文档包含与用户问题相关的关键词或语义信息，即判定为相关。
        请仅返回二元评分：'yes'（相关）或'no'（不相关），无需额外解释。""",
        input_variables=["context", "question"],  # 提示模板的输入变量（文档内容和用户问题）
    )

    # 构建相关性判断链（提示模板 → 结构化输出模型）
    chain = prompt | llm_with_tool

    # 从状态中提取消息历史和关键信息
    messages = state["messages"]  # 所有消息历史
    last_message = messages[-1]  # 最后一条消息（通常是检索到的文档内容）

    question = messages[0].content  # 第一条消息（用户的原始问题）
    docs = last_message.content  # 检索到的文档内容

    # 调用判断链，获取相关性评分结果
    scored_result = chain.invoke({"question": question, "context": docs})

    score = scored_result.binary_score  # 提取二元评分结果

    # 根据评分结果决定流程走向
    if score == "yes":
        print("---决策结果：文档相关，进入回答生成阶段---")
        return "generate"  # 文档相关，跳转到"生成回答"节点
    else:
        print("---决策结果：文档不相关，进入问题重写阶段---")
        print(f"评分结果：{score}")
        return "rewrite"  # 文档不相关，跳转到"重写问题"节点


# 定义LangGraph的节点（Node）：用于执行具体逻辑
## 智能体节点（核心决策节点）
def agent(state, tools):
    """
    智能体核心节点：根据当前状态（用户问题/历史消息）决定是否调用检索工具，或直接结束流程。

    参数：
        state (dict): 当前智能体状态（包含消息历史）
        tools (list): 智能体可调用的工具列表（此处为检索工具）

    返回：
        dict: 更新后的状态（追加智能体的决策消息）
    """
    print("---调用智能体核心节点---")
    messages = state["messages"]  # 提取消息历史

    # 初始化谷歌生成式AI聊天模型（用于智能体决策）
    model = ChatGoogleGenerativeAI(
        api_key=st.session_state.gemini_api_key,
        temperature=0,  # 低温确保决策逻辑稳定
        streaming=True,
        model="gemini-2.0-flash"
    )

    # 为模型绑定工具（告知模型可调用的工具及用途）
    model = model.bind_tools(tools)

    # 调用模型，获取智能体的决策（是否调用工具）
    response = model.invoke(messages)

    # 返回更新后的状态（将智能体决策消息追加到消息历史）
    return {"messages": [response]}


## 问题重写节点
def rewrite(state):
    """
    问题重写节点：当检索到的文档不相关时，优化用户问题以提高后续检索准确性。

    参数：
        state (dict): 当前智能体状态（包含消息历史）

    返回：
        dict: 更新后的状态（追加重写后的问题）
    """

    print("---进入问题重写阶段---")
    messages = state["messages"]  # 提取消息历史
    question = messages[0].content  # 提取用户原始问题

    # 构建问题重写的提示（引导模型理解原始问题语义并优化表述）
    msg = [
        HumanMessage(
            content=f"""请分析用户原始问题的核心语义意图，生成更精准的优化问题。
                    原始问题如下：
                    \n ------- \n
                    {question} 
                    \n ------- \n
                    优化要求：
                    1. 保留原始问题的核心需求；
                    2. 修正模糊表述（如歧义词汇、不完整表述）；
                    3. 补充必要上下文（若原始问题缺少关键信息）；
                    4. 仅返回优化后的问题，无需额外解释。""",
        )
    ]

    # 初始化用于重写问题的模型（低温确保表述准确）
    model = ChatGoogleGenerativeAI(
        api_key=st.session_state.gemini_api_key,
        temperature=0,
        model="gemini-2.0-flash",
        streaming=True
    )

    # 调用模型生成重写后的问题
    response = model.invoke(msg)

    # 返回更新后的状态（将重写后的问题追加到消息历史）
    return {"messages": [response]}


## 回答生成节点
def generate(state):
    """
    回答生成节点：当检索到相关文档后，结合文档内容生成用户问题的最终回答。

    参数：
        state (dict): 当前智能体状态（包含消息历史）

    返回：
        dict: 更新后的状态（追加生成的最终回答）
    """
    print("---进入回答生成阶段---")
    messages = state["messages"]  # 提取消息历史
    question = messages[0].content  # 提取用户原始问题
    last_message = messages[-1]  # 提取最后一条消息（检索到的相关文档）

    docs = last_message.content  # 提取相关文档内容

    # 从LangChain中心库拉取预置的RAG提示模板（优化回答生成逻辑）
    prompt_template = hub.pull("rlm/rag-prompt")

    # 初始化用于生成回答的聊天模型
    chat_model = ChatGoogleGenerativeAI(
        api_key=st.session_state.gemini_api_key,
        model="gemini-2.0-flash",
        temperature=0,  # 低温确保回答基于文档事实，不编造信息
        streaming=True
    )

    # 初始化字符串输出解析器（将模型输出转换为纯文本）
    output_parser = StrOutputParser()

    # 构建RAG回答生成链（提示模板 → 聊天模型 → 输出解析器）
    rag_chain = prompt_template | chat_model | output_parser

    # 调用RAG链，结合文档和问题生成最终回答
    response = rag_chain.invoke({"context": docs, "question": question})

    # 返回更新后的状态（将最终回答追加到消息历史）
    return {"messages": [response]}


# 构建LangGraph状态图的函数
def get_graph(retriever_tool):
    tools = [retriever_tool]  # 构建工具列表（此处仅包含检索工具）

    # 初始化状态图（基于AgentState定义的状态结构）
    workflow = StateGraph(AgentState)

    # 添加"智能体"节点：使用偏函数固定tools参数（避免重复传递）
    workflow.add_node("agent", partial(agent, tools=tools))

    # 添加"检索"节点：使用LangGraph预置的ToolNode（自动处理工具调用逻辑）
    retrieve = ToolNode(tools)
    workflow.add_node("retrieve", retrieve)

    # 添加"问题重写"节点
    workflow.add_node("rewrite", rewrite)

    # 添加"回答生成"节点（确认文档相关后生成最终回答）
    workflow.add_node("generate", generate)

    # 设置状态图的起始节点：从"智能体"节点开始流程
    workflow.add_edge(START, "agent")

    # 添加"智能体"节点的条件跳转逻辑（决定是否调用检索工具）
    workflow.add_conditional_edges(
        "agent",  # 源节点（从"智能体"节点跳转）
        tools_condition,  # 条件判断函数（LangGraph预置：判断是否调用工具）
        {
            # 条件结果与目标节点的映射：
            "tools": "retrieve",  # 若需调用工具，跳转到"检索"节点
            END: END  # 若无需调用工具，直接结束流程
        },
    )

    # 添加"检索"节点的条件跳转逻辑（根据文档相关性决定后续流程）
    workflow.add_conditional_edges(
        "retrieve",  # 源节点（从"检索"节点跳转）
        grade_documents,  # 条件判断函数（自定义：文档相关性判断）
    )

    # 添加"回答生成"节点的跳转逻辑：生成回答后直接结束流程
    workflow.add_edge("generate", END)

    # 添加"问题重写"节点的跳转逻辑：重写问题后返回"智能体"节点重新决策
    workflow.add_edge("rewrite", "agent")

    # 编译状态图（生成可执行的图结构）
    graph = workflow.compile()

    return graph  # 返回编译完成的状态图


# 从状态图流式输出中提取最终回答的函数
def generate_message(graph, inputs):
    generated_message = ""  # 初始化最终回答变量

    # 流式遍历状态图的输出（逐步获取每个节点的执行结果）
    for output in graph.stream(inputs):
        # 遍历输出中的键值对（键为节点名，值为节点执行结果）
        for key, value in output.items():
            # 若节点为"generate"（回答生成节点）且结果为字典类型
            if key == "generate" and isinstance(value, dict):
                # 提取"generate"节点生成的回答（取第一条消息）
                generated_message = value.get("messages", [""])[0]

    return generated_message  # 返回最终回答


def add_documents_to_qdrant(url, db):
    try:
        # 加载网页文档
        docs = WebBaseLoader(url).load()
        # 初始化递归字符文本分割器（基于tiktoken编码器）
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=100,  # 文本块大小
            chunk_overlap=50  # 文本块重叠部分大小
        )
        # 将文档分割为多个文本块
        doc_chunks = text_splitter.split_documents(docs)
        # 为每个文本块生成唯一ID
        uuids = [str(uuid4()) for _ in range(len(doc_chunks))]
        # 将文本块添加到向量数据库
        db.add_documents(documents=doc_chunks, ids=uuids)
        return True
    except Exception as e:
        # 显示添加文档时的错误信息
        st.error(f"添加文档出错: {str(e)}")
        return False

def main():
    # 设置侧边栏
    set_sidebar()

    # 检查API密钥是否已设置
    if not all([st.session_state.qdrant_host,
                st.session_state.qdrant_api_key,
                st.session_state.gemini_api_key]):
        st.warning("请先在侧边栏配置您的API密钥")
        return

    # 初始化组件
    embedding_model, client, db = initialize_components()
    if not all([embedding_model, client, db]):
        return

    # 初始化检索器和工具
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})  # 相似性检索，返回前5个结果
    retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_blog_posts",  # 工具名称
        "搜索并返回有关LLM（大语言模型）、LLM智能体、提示工程和LLM对抗性攻击的博客文章信息。"  # 工具描述
    )
    tools = [retriever_tool]  # 工具列表

    # URL输入部分
    url = st.text_input(
        ":link: 粘贴博客链接:",
        placeholder="例如: https://lilianweng.github.io/posts/2023-06-23-agent/"
    )
    if st.button("输入URL"):
        if url:
            with st.spinner("正在处理文档..."):
                # 将文档添加到向量数据库
                if add_documents_to_qdrant(url, db):
                    st.success("文档添加成功!")
                else:
                    st.error("文档添加失败")
        else:
            st.warning("请输入URL")

    # 查询部分
    graph = get_graph(retriever_tool)  # 获取状态图
    query = st.text_area(
        ":bulb: 输入关于博客文章的查询:",
        placeholder="例如: 莉莲·翁（Lilian Weng）关于智能体记忆类型说了些什么?"
    )

    if st.button("提交查询"):
        if not query:
            st.warning("请输入查询内容")
            return

        # 准备输入消息
        inputs = {"messages": [HumanMessage(content=query)]}
        with st.spinner("正在生成回复..."):
            try:
                # 生成回复
                response = generate_message(graph, inputs)
                st.write(response)
            except Exception as e:
                st.error(f"生成回复出错: {str(e)}")

    # 页脚信息
    st.markdown("---")
    st.write("使用 :blue-background[LangChain] | :blue-background[LangGraph] 构建，作者: [Charan](https://www.linkedin.com/in/codewithcharan/)")

if __name__ == "__main__":
    main()