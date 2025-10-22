import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain import hub
import tempfile
from langgraph.prebuilt import create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
from typing import TypedDict, List
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from time import sleep
from tenacity import retry, wait_exponential, stop_after_attempt


def init_session_state():
    """初始化会话状态变量"""
    if 'api_keys_submitted' not in st.session_state:
        st.session_state.api_keys_submitted = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'qdrant_api_key' not in st.session_state:
        st.session_state.qdrant_api_key = ""
    if 'qdrant_url' not in st.session_state:
        st.session_state.qdrant_url = ""


def sidebar_api_form():
    """侧边栏API凭证表单"""
    with st.sidebar:
        st.header("API凭证")

        if st.session_state.api_keys_submitted:
            st.success("API凭证已验证")
            if st.button("重置凭证"):
                st.session_state.clear()
                st.rerun()
            return True

        with st.form("api_credentials"):
            cohere_key = st.text_input("Cohere API密钥", type="password")
            qdrant_key = st.text_input("Qdrant API密钥", type="password", help="输入您的Qdrant API密钥")
            qdrant_url = st.text_input("Qdrant URL",
                                       placeholder="https://xyz-example.eu-central.aws.cloud.qdrant.io:6333",
                                       help="输入您的Qdrant实例URL")

            if st.form_submit_button("提交凭证"):
                try:
                    client = QdrantClient(url=qdrant_url, api_key=qdrant_key, timeout=60)
                    client.get_collections()

                    st.session_state.cohere_api_key = cohere_key
                    st.session_state.qdrant_api_key = qdrant_key
                    st.session_state.qdrant_url = qdrant_url
                    st.session_state.api_keys_submitted = True

                    st.success("凭证验证成功！")
                    st.rerun()
                except Exception as e:
                    st.error(f"Qdrant连接失败: {str(e)}")
        return False


def init_qdrant() -> QdrantClient:
    """初始化Qdrant客户端"""
    if not st.session_state.get("qdrant_api_key"):
        raise ValueError("未提供Qdrant API密钥")
    if not st.session_state.get("qdrant_url"):
        raise ValueError("未提供Qdrant URL")

    return QdrantClient(url=st.session_state.qdrant_url,
                        api_key=st.session_state.qdrant_api_key,
                        timeout=60)


# 初始化会话状态
init_session_state()

# 检查API凭证是否提交
if not sidebar_api_form():
    st.info("请在侧边栏输入您的API凭证以继续。")
    st.stop()

# 初始化嵌入模型和聊天模型
embedding = CohereEmbeddings(model="embed-english-v3.0",
                             cohere_api_key=st.session_state.cohere_api_key)

chat_model = ChatCohere(model="command-r7b-12-2024",
                        temperature=0.1,
                        max_tokens=512,
                        verbose=True,
                        cohere_api_key=st.session_state.cohere_api_key)

# 初始化Qdrant客户端
client = init_qdrant()


def process_document(file):
    """处理上传的文档并分割为文本块"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)

        os.unlink(tmp_path)

        return texts
    except Exception as e:
        st.error(f"处理文档时出错: {e}")
        return []


COLLECTION_NAME = "cohere_rag"


def create_vector_stores(texts):
    """创建并填充向量存储"""
    try:
        try:
            client.create_collection(collection_name=COLLECTION_NAME,
                                     vectors_config=VectorParams(size=1024,
                                                                 distance=Distance.COSINE))
            st.success(f"已创建新集合: {COLLECTION_NAME}")
        except Exception as e:
            if "already exists" not in str(e).lower():
                raise e

        vector_store = QdrantVectorStore(client=client,
                                         collection_name=COLLECTION_NAME,
                                         embedding=embedding)

        with st.spinner('正在Qdrant中存储文档...'):
            vector_store.add_documents(texts)
            st.success("文档已成功存储到Qdrant中！")

        return vector_store

    except Exception as e:
        st.error(f"创建向量存储时出错: {str(e)}")
        return None


# 定义代理状态模式
class AgentState(TypedDict):
    """代理的状态模式"""
    messages: List[HumanMessage | AIMessage | SystemMessage]
    is_last_step: bool


class RateLimitedDuckDuckGo(DuckDuckGoSearchRun):
    """带速率限制的DuckDuckGo搜索工具"""

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10),
           stop=stop_after_attempt(3))
    def run(self, query: str) -> str:
        """带速率限制的搜索运行方法"""
        try:
            sleep(2)  # 请求之间添加延迟
            return super().run(query)
        except Exception as e:
            if "Ratelimit" in str(e):
                sleep(5)  # 遇到速率限制时延长延迟
                return super().run(query)
            raise e


def create_fallback_agent(chat_model: BaseLanguageModel):
    """创建用于网络研究的LangGraph代理"""

    def web_research(query: str) -> str:
        """带结果格式化的网络搜索"""
        try:
            search = DuckDuckGoSearchRun(num_results=5)
            results = search.run(query)
            return results
        except Exception as e:
            return f"搜索失败: {str(e)}。基于常识提供答案。"

    tools = [web_research]

    agent = create_react_agent(model=chat_model,
                               tools=tools,
                               debug=False)

    return agent


def process_query(vectorstore, query) -> tuple[str, list]:
    """使用RAG处理查询，网络搜索作为备用方案"""
    try:
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 10,
                "score_threshold": 0.7
            }
        )

        relevant_docs = retriever.get_relevant_documents(query)

        if relevant_docs:
            retrieval_qa_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
            combine_docs_chain = create_stuff_documents_chain(chat_model, retrieval_qa_prompt)
            retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
            response = retrieval_chain.invoke({"input": query})
            return response['answer'], relevant_docs

        else:
            st.info("未找到相关文档。正在搜索网络...")
            fallback_agent = create_fallback_agent(chat_model)

            with st.spinner('正在研究...'):
                agent_input = {
                    "messages": [
                        HumanMessage(content=f"""请彻底研究以下问题: '{query}' 并提供详细全面的回答。确保从可信来源收集最新信息。至少400字。""")
                    ],
                    "is_last_step": False
                }

                config = {"recursion_limit": 100}

                try:
                    response = fallback_agent.invoke(agent_input, config=config)

                    if isinstance(response, dict) and "messages" in response:
                        last_message = response["messages"][-1]
                        answer = last_message.content if hasattr(last_message, 'content') else str(last_message)

                        return f"""网络搜索结果:
{answer}
""", []

                except Exception as agent_error:
                    fallback_response = chat_model.invoke(f"请提供关于以下内容的一般性回答: {query}").content
                    return f"网络搜索不可用。一般性回答: {fallback_response}", []

    except Exception as e:
        st.error(f"错误: {str(e)}")
        return "处理过程中遇到错误。请尝试重新表述您的问题。", []


def post_process(answer, sources):
    """后处理答案并格式化来源"""
    answer = answer.strip()

    # 总结长答案
    if len(answer) > 500:
        summary_prompt = f"用2-3句话总结以下答案: {answer}"
        summary = chat_model.invoke(summary_prompt).content
        answer = f"{summary}\n\n完整答案: {answer}"

    formatted_sources = []
    for i, source in enumerate(sources, 1):
        formatted_source = f"{i}. {source.page_content[:200]}..."
        formatted_sources.append(formatted_source)
    return answer, formatted_sources


# 主界面
st.title("带Cohere ⌘R的RAG代理")

uploaded_file = st.file_uploader("选择PDF或图片文件", type=["pdf", "jpg", "jpeg"])

if uploaded_file is not None and 'processed_file' not in st.session_state:
    with st.spinner('正在处理文件... 图片处理可能需要较长时间。'):
        texts = process_document(uploaded_file)
        vectorstore = create_vector_stores(texts)
        if vectorstore:
            st.session_state.vectorstore = vectorstore
            st.session_state.processed_file = True
            st.success('文件已上传并处理成功！')
        else:
            st.error('文件处理失败。请重试。')

# 显示聊天历史
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 处理用户查询
if query := st.chat_input("询问关于文档的问题:"):
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    if st.session_state.vectorstore:
        with st.chat_message("assistant"):
            try:
                answer, sources = process_query(st.session_state.vectorstore, query)
                st.markdown(answer)

                if sources:
                    with st.expander("来源"):
                        for source in sources:
                            st.markdown(f"- {source.page_content[:200]}...")

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:
                st.error(f"错误: {str(e)}")
                st.info("请尝试再次提问。")
    else:
        st.error("请先上传文档。")

# 侧边栏附加功能
with st.sidebar:
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button('清除聊天历史'):
            st.session_state.chat_history = []
            st.rerun()
    with col2:
        if st.button('清除所有数据'):
            try:
                collections = client.get_collections().collections
                collection_names = [col.name for col in collections]

                if COLLECTION_NAME in collection_names:
                    client.delete_collection(COLLECTION_NAME)
                if f"{COLLECTION_NAME}_compressed" in collection_names:
                    client.delete_collection(f"{COLLECTION_NAME}_compressed")

                st.session_state.vectorstore = None
                st.session_state.chat_history = []
                st.success("所有数据已成功清除！")
                st.rerun()
            except Exception as e:
                st.error(f"清除数据时出错: {str(e)}")