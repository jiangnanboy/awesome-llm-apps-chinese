from langchain import hub
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from pydantic import BaseModel, Field
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_community.tools import TavilySearchResults
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from typing import Dict, TypedDict
from langchain_core.prompts import PromptTemplate
import pprint
import yaml
import nest_asyncio
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import tempfile
import os
from langchain_anthropic import ChatAnthropic
from tenacity import retry, stop_after_attempt, wait_exponential

nest_asyncio.apply()

retriever = None


def initialize_session_state():
    """初始化会话状态变量，用于存储API密钥和URL"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        # 初始化API密钥和URL
        st.session_state.anthropic_api_key = ""
        st.session_state.openai_api_key = ""
        st.session_state.tavily_api_key = ""
        st.session_state.qdrant_api_key = ""
        st.session_state.qdrant_url = "http://localhost:6333"
        st.session_state.doc_url = "https://arxiv.org/pdf/2307.09288.pdf"


def setup_sidebar():
    """设置侧边栏，用于输入API密钥和配置信息"""
    with st.sidebar:
        st.subheader("API配置")
        st.session_state.anthropic_api_key = st.text_input("Anthropic API密钥", value=st.session_state.anthropic_api_key,
                                                           type="password", help="Claude 3模型所需")
        st.session_state.openai_api_key = st.text_input("OpenAI API密钥", value=st.session_state.openai_api_key,
                                                        type="password")
        st.session_state.tavily_api_key = st.text_input("Tavily API密钥", value=st.session_state.tavily_api_key,
                                                        type="password")
        st.session_state.qdrant_url = st.text_input("Qdrant URL", value=st.session_state.qdrant_url)
        st.session_state.qdrant_api_key = st.text_input("Qdrant API密钥", value=st.session_state.qdrant_api_key,
                                                        type="password")
        st.session_state.doc_url = st.text_input("文档URL", value=st.session_state.doc_url)

        if not all([st.session_state.openai_api_key, st.session_state.anthropic_api_key, st.session_state.qdrant_url]):
            st.warning("请提供所需的API密钥和URL")
            st.stop()

        st.session_state.initialized = True


initialize_session_state()
setup_sidebar()

# 使用会话状态变量而非配置
openai_api_key = st.session_state.openai_api_key
tavily_api_key = st.session_state.tavily_api_key
anthropic_api_key = st.session_state.anthropic_api_key

# 更新嵌入初始化
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=st.session_state.openai_api_key
)

# 更新Qdrant客户端初始化
client = QdrantClient(
    url=st.session_state.qdrant_url,
    api_key=st.session_state.qdrant_api_key
)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def execute_tavily_search(tool, query):
    return tool.invoke({"query": query})


def web_search(state):
    """基于重新表述的问题使用Tavily API进行网络搜索"""
    print("~-网络搜索-~")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]

    # 创建进度占位符
    progress_placeholder = st.empty()
    progress_placeholder.info("正在初始化网络搜索...")

    try:
        # 验证Tavily API密钥
        if not st.session_state.tavily_api_key:
            progress_placeholder.warning("未提供Tavily API密钥 - 跳过网络搜索")
            return {"keys": {"documents": documents, "question": question}}

        progress_placeholder.info("正在配置搜索工具...")

        # 初始化Tavily搜索工具
        tool = TavilySearchResults(
            api_key=st.session_state.tavily_api_key,
            max_results=3,
            search_depth="advanced"
        )

        # 使用重试逻辑执行搜索
        progress_placeholder.info("正在执行搜索查询...")
        try:
            search_results = execute_tavily_search(tool, question)
        except Exception as search_error:
            progress_placeholder.error(f"多次尝试后搜索失败: {str(search_error)}")
            return {"keys": {"documents": documents, "question": question}}

        if not search_results:
            progress_placeholder.warning("未找到搜索结果")
            return {"keys": {"documents": documents, "question": question}}

        # 处理结果
        progress_placeholder.info("正在处理搜索结果...")
        web_results = []
        for result in search_results:
            # 提取并格式化相关信息
            content = (
                f"标题: {result.get('title', '无标题')}\n"
                f"内容: {result.get('content', '无内容')}\n"
            )
            web_results.append(content)

        # 从结果创建文档
        web_document = Document(
            page_content="\n\n".join(web_results),
            metadata={
                "来源": "tavily_search",
                "查询": question,
                "结果数量": len(web_results)
            }
        )
        documents.append(web_document)

        progress_placeholder.success(f"成功添加 {len(web_results)} 条搜索结果")

    except Exception as error:
        error_msg = f"网络搜索错误: {str(error)}"
        print(error_msg)
        progress_placeholder.error(error_msg)
    finally:
        progress_placeholder.empty()
    return {"keys": {"documents": documents, "question": question}}


def load_documents(file_or_url: str, is_url: bool = True) -> list:
    try:
        if is_url:
            loader = WebBaseLoader(file_or_url)
            loader.requests_per_second = 1
        else:
            file_extension = os.path.splitext(file_or_url)[1].lower()
            if file_extension == '.pdf':
                loader = PyPDFLoader(file_or_url)
            elif file_extension in ['.txt', '.md']:
                loader = TextLoader(file_or_url)
            else:
                raise ValueError(f"不支持的文件类型: {file_extension}")

        return loader.load()
    except Exception as e:
        st.error(f"加载文档时出错: {str(e)}")
        return []


st.subheader("文档输入")
input_option = st.radio("选择输入方式:", ["URL", "文件上传"])

docs = None

if input_option == "URL":
    url = st.text_input("输入文档URL:", value=st.session_state.doc_url)
    if url:
        docs = load_documents(url, is_url=True)
else:
    uploaded_file = st.file_uploader("上传文档", type=['pdf', 'txt', 'md'])
    if uploaded_file:
        # 创建临时文件存储上传内容
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            docs = load_documents(tmp_file.name, is_url=False)
        # 清理临时文件
        os.unlink(tmp_file.name)

if docs:
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=100
    )
    all_splits = text_splitter.split_documents(docs)

    client = QdrantClient(url=st.session_state.qdrant_url, api_key=st.session_state.qdrant_api_key)
    collection_name = "rag-qdrant"

    try:
        # 如果集合存在，尝试删除它
        client.delete_collection(collection_name)
    except Exception:
        pass

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )

    # 创建向量存储
    vectorstore = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embeddings,
    )

    # 将文档添加到向量存储
    vectorstore.add_documents(all_splits)
    retriever = vectorstore.as_retriever()


class GraphState(TypedDict):
    keys: Dict[str, any]


def retrieve(state):
    print("~-检索-~")
    state_dict = state["keys"]
    question = state_dict["question"]

    if retriever is None:
        return {"keys": {"documents": [], "question": question}}

    documents = retriever.get_relevant_documents(question)
    return {"keys": {"documents": documents, "question": question}}


def generate(state):
    """使用Claude 3模型生成答案"""
    print("~-生成-~")
    state_dict = state["keys"]
    question, documents = state_dict["question"], state_dict["documents"]
    try:
        prompt = PromptTemplate(template="""基于以下上下文，请回答问题。
            上下文: {context}
            问题: {question}
            回答:""", input_variables=["context", "question"])
        llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", api_key=st.session_state.anthropic_api_key,
                            temperature=0, max_tokens=1000)
        context = "\n\n".join(doc.page_content for doc in documents)

        # 创建并运行链
        rag_chain = (
                {"context": lambda x: context, "question": lambda x: question}
                | prompt
                | llm
                | StrOutputParser()
        )

        generation = rag_chain.invoke({})

        return {
            "keys": {
                "documents": documents,
                "question": question,
                "generation": generation
            }
        }

    except Exception as e:
        error_msg = f"生成函数出错: {str(e)}"
        print(error_msg)
        st.error(error_msg)
        return {"keys": {"documents": documents, "question": question,
                         "generation": "抱歉，生成回答时遇到错误。"}}


def grade_documents(state):
    """判断检索到的文档是否相关"""
    print("~-检查相关性-~")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]

    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", api_key=st.session_state.anthropic_api_key,
                        temperature=0, max_tokens=1000)

    prompt = PromptTemplate(template="""你需要评估检索到的文档与用户问题的相关性。
        只返回一个JSON对象，包含"score"字段，值为"yes"或"no"。
        不要包含任何其他文本或解释。

        文档: {context}
        问题: {question}

        规则:
        - 检查相关关键词或语义含义
        - 使用宽松的评分标准，只过滤明显不匹配的内容
        - 严格按照示例格式返回: {{"score": "yes"}} 或 {{"score": "no"}}""",
                            input_variables=["context", "question"])

    chain = (
            prompt
            | llm
            | StrOutputParser()
    )

    filtered_docs = []
    search = "No"

    for d in documents:
        try:
            response = chain.invoke({"question": question, "context": d.page_content})
            import re
            json_match = re.search(r'\{.*\}', response)
            if json_match:
                response = json_match.group()

            import json
            score = json.loads(response)

            if score.get("score") == "yes":
                print("~-评分: 文档相关-~")
                filtered_docs.append(d)
            else:
                print("~-评分: 文档不相关-~")
                search = "Yes"

        except Exception as e:
            print(f"评估文档时出错: {str(e)}")
            # 出错时保留文档以确保安全
            filtered_docs.append(d)
            continue

    return {"keys": {"documents": filtered_docs, "question": question, "run_web_search": search}}


def transform_query(state):
    """转换查询以生成更好的问题"""
    print("~-转换查询-~")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]

    # 创建提示模板
    prompt = PromptTemplate(
        template="""通过分析问题的核心语义和意图，生成一个优化的搜索版本。
        \n ------- \n
        {question}
        \n ------- \n
        只返回改进后的问题，不要附加任何其他文本:""",
        input_variables=["question"],
    )

    # 使用Claude而非Gemini
    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20240620",
        anthropic_api_key=st.session_state.anthropic_api_key,
        temperature=0,
        max_tokens=1000
    )

    # 提示
    chain = prompt | llm | StrOutputParser()
    better_question = chain.invoke({"question": question})

    return {
        "keys": {"documents": documents, "question": better_question}
    }


def decide_to_generate(state):
    print("~-决定生成-~")
    state_dict = state["keys"]
    search = state_dict["run_web_search"]

    if search == "Yes":

        print("~-决定: 转换查询并运行网络搜索-~")
        return "transform_query"
    else:
        print("~-决定: 生成-~")
        return "generate"


def format_document(doc: Document) -> str:
    return f"""
    来源: {doc.metadata.get('source', '未知')}
    标题: {doc.metadata.get('title', '无标题')}
    内容: {doc.page_content[:200]}...
    """


def format_state(state: dict) -> str:
    formatted = {}

    for key, value in state.items():
        if key == "documents":
            formatted[key] = [format_document(doc) for doc in value]
        else:
            formatted[key] = value

    return formatted


workflow = StateGraph(GraphState)

# 通过langgraph定义节点
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)
workflow.add_node("web_search", web_search)

# 构建图
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "web_search")
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()

st.title("🔄 校正型RAG代理")

st.text("一个示例查询: 这篇研究论文中的实验结果和消融研究是什么?")

# 用户输入
user_question = st.text_input("请输入你的问题:")

if user_question:
    inputs = {
        "keys": {
            "question": user_question,
        }
    }

    for output in app.stream(inputs):
        for key, value in output.items():
            with st.expander(f"步骤 '{key}':"):
                st.text(pprint.pformat(format_state(value["keys"]), indent=2, width=80))

    final_generation = value['keys'].get('generation', '未生成最终回答。')
    st.subheader("最终回答:")
    st.write(final_generation)