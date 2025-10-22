import os
import logging
import streamlit as st
from raglite import RAGLiteConfig, insert_document, hybrid_search, retrieve_chunks, rerank_chunks, rag
from rerankers import Reranker
from typing import List
from pathlib import Path
import anthropic
import time
import warnings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message=".*torch.classes.*")

RAG_SYSTEM_PROMPT = """
你是一个友好且知识渊博的助手，能提供完整且有洞察力的答案。
仅使用下面的上下文来回答用户的问题。
回答时，严禁直接或间接提及上下文的存在。
相反，你必须将上下文的内容视为完全是你工作记忆的一部分。
""".strip()


def initialize_config(openai_key: str, anthropic_key: str, cohere_key: str, db_url: str) -> RAGLiteConfig:
    """初始化并返回一个带有指定API密钥和数据库URL的RAGLiteConfig对象。

    此函数将提供的API密钥设置在环境变量中，并返回一个
    用给定的数据库URL和预定义的语言模型、嵌入器和重排序器设置配置的RAGLiteConfig对象。

    参数:
        openai_key (str): OpenAI服务的API密钥。
        anthropic_key (str): Anthropic服务的API密钥。
        cohere_key (str): Cohere服务的API密钥。
        db_url (str): 用于连接到所需数据源的数据库URL。

    返回:
        RAGLiteConfig: 用指定参数初始化的配置对象。

    异常:
        ValueError: 如果在设置配置时出现问题，将引发带有详细信息的错误。"""
    try:
        os.environ["OPENAI_API_KEY"] = openai_key
        os.environ["ANTHROPIC_API_KEY"] = anthropic_key
        os.environ["COHERE_API_KEY"] = cohere_key

        return RAGLiteConfig(
            db_url=db_url,
            llm="claude-3-opus-20240229",
            embedder="text-embedding-3-large",
            embedder_normalize=True,
            chunk_max_size=2000,
            embedder_sentence_window_size=2,
            reranker=Reranker("cohere", api_key=cohere_key, lang="en")
        )
    except Exception as e:
        raise ValueError(f"配置错误: {e}")


def process_document(file_path: str) -> bool:
    """通过将文档插入到具有给定配置的系统中来处理文档。

    此函数检查会话状态中是否初始化了配置。
    如果配置存在，它会尝试使用此配置插入位于给定文件路径的文档。

    参数:
        file_path (str): 要处理的文档的路径。

    返回:
        bool: 如果文档成功处理则为True；否则为False。"""
    try:
        if not st.session_state.get('my_config'):
            raise ValueError("配置未初始化")
        insert_document(Path(file_path), config=st.session_state.my_config)
        return True
    except Exception as e:
        logger.error(f"处理文档时出错: {str(e)}")
        return False


def perform_search(query: str) -> List[dict]:
    """执行混合搜索并返回基于查询的排序后的块列表。

    此函数使用混合搜索方法执行搜索，检索相关块，并根据查询对它们进行重排序。
    它处理过程中发生的任何异常并记录错误。

    参数:
        query (str): 搜索查询字符串。

    返回:
        List[dict]: 表示排序后块的字典列表。如果未找到结果或发生错误，则返回空列表。"""
    try:
        chunk_ids, scores = hybrid_search(query, num_results=10, config=st.session_state.my_config)
        if not chunk_ids:
            return []
        chunks = retrieve_chunks(chunk_ids, config=st.session_state.my_config)
        return rerank_chunks(query, chunks, config=st.session_state.my_config)
    except Exception as e:
        logger.error(f"搜索错误: {str(e)}")
        return []


def handle_fallback(query: str) -> str:
    try:
        client = anthropic.Anthropic(api_key=st.session_state.user_env["ANTHROPIC_API_KEY"])
        system_prompt = """你是一个有帮助的AI助手。当你不知道某些事情时，要诚实地说明。
        提供清晰、简洁和准确的回应。如果问题与任何特定文档无关，请使用你的常识来回答。"""

        message = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": query}],
            temperature=0.7
        )
        return message.content[0].text
    except Exception as e:
        logger.error(f"备用处理错误: {str(e)}")
        st.error(f"备用处理错误: {str(e)}")  # 在UI中显示错误
        return "抱歉，处理你的请求时遇到错误。请重试。"


def main():
    st.set_page_config(page_title="LLM驱动的混合搜索-RAG助手", layout="wide")

    for state_var in ['chat_history', 'documents_loaded', 'my_config', 'user_env']:
        if state_var not in st.session_state:
            st.session_state[
                state_var] = [] if state_var == 'chat_history' else False if state_var == 'documents_loaded' else None if state_var == 'my_config' else {}

    with st.sidebar:
        st.title("配置")
        openai_key = st.text_input("OpenAI API密钥", value=st.session_state.get('openai_key', ''), type="password",
                                   placeholder="sk-...")
        anthropic_key = st.text_input("Anthropic API密钥", value=st.session_state.get('anthropic_key', ''),
                                      type="password", placeholder="sk-ant-...")
        cohere_key = st.text_input("Cohere API密钥", value=st.session_state.get('cohere_key', ''), type="password",
                                   placeholder="输入Cohere密钥")
        db_url = st.text_input("数据库URL", value=st.session_state.get('db_url', 'sqlite:///raglite.sqlite'),
                               placeholder="sqlite:///raglite.sqlite")

        if st.button("保存配置"):
            try:
                if not all([openai_key, anthropic_key, cohere_key, db_url]):
                    st.error("所有字段都是必填的！")
                    return

                for key, value in {'openai_key': openai_key, 'anthropic_key': anthropic_key, 'cohere_key': cohere_key,
                                   'db_url': db_url}.items():
                    st.session_state[key] = value

                st.session_state.my_config = initialize_config(openai_key=openai_key, anthropic_key=anthropic_key,
                                                               cohere_key=cohere_key, db_url=db_url)
                st.session_state.user_env = {"ANTHROPIC_API_KEY": anthropic_key}
                st.success("配置保存成功！")
            except Exception as e:
                st.error(f"配置错误: {str(e)}")

    st.title("👀 带有混合搜索的RAG应用")

    if st.session_state.my_config:
        uploaded_files = st.file_uploader("上传PDF文档", type=["pdf"], accept_multiple_files=True, key="pdf_uploader")

        if uploaded_files:
            success = False
            for uploaded_file in uploaded_files:
                with st.spinner(f"正在处理 {uploaded_file.name}..."):
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getvalue())

                    if process_document(temp_path):
                        st.success(f"成功处理: {uploaded_file.name}")
                        success = True
                    else:
                        st.error(f"处理失败: {uploaded_file.name}")
                    os.remove(temp_path)

            if success:
                st.session_state.documents_loaded = True
                st.success("文档已准备就绪！现在你可以询问有关它们的问题了。")

    if st.session_state.documents_loaded:
        for msg in st.session_state.chat_history:
            with st.chat_message("user"): st.write(msg[0])
            with st.chat_message("assistant"): st.write(msg[1])

        user_input = st.chat_input("询问有关文档的问题...")
        if user_input:
            with st.chat_message("user"):
                st.write(user_input)
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                try:
                    reranked_chunks = perform_search(query=user_input)
                    if not reranked_chunks or len(reranked_chunks) == 0:
                        logger.info("未找到相关文档。切换到Claude。")
                        st.info("未找到相关文档。使用常识来回答。")
                        full_response = handle_fallback(user_input)
                    else:
                        formatted_messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": msg}
                                              for i, msg in
                                              enumerate([m for pair in st.session_state.chat_history for m in pair]) if
                                              msg]

                        response_stream = rag(prompt=user_input,
                                              system_prompt=RAG_SYSTEM_PROMPT,
                                              search=hybrid_search,
                                              messages=formatted_messages,
                                              max_contexts=5,
                                              config=st.session_state.my_config)

                        full_response = ""
                        for chunk in response_stream:
                            full_response += chunk
                            message_placeholder.markdown(full_response + "▌")

                    message_placeholder.markdown(full_response)
                    st.session_state.chat_history.append((user_input, full_response))
                except Exception as e:
                    st.error(f"错误: {str(e)}")
    else:
        st.info("请配置你的API密钥并上传文档以开始使用。" if not st.session_state.my_config else "请上传一些文档以开始使用。")


if __name__ == "__main__":
    main()