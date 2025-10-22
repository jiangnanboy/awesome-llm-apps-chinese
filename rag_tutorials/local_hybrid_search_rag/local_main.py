import os
import logging
import streamlit as st
from raglite import RAGLiteConfig, insert_document, hybrid_search, retrieve_chunks, rerank_chunks, rag
from rerankers import Reranker
from typing import List, Dict, Any
from pathlib import Path
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


def initialize_config(settings: Dict[str, Any]) -> RAGLiteConfig:
    """基于提供的设置初始化并返回RAGLiteConfig对象。

    此函数使用`settings`字典中指定的数据库URL、语言模型路径和嵌入器路径构建RAGLiteConfig对象。
    配置包括嵌入器归一化和块大小的默认选项。重排序器也使用预定义模型初始化。

    参数:
        settings (Dict[str, Any]): 包含配置参数的字典。预期的键是'DBUrl'、'LLMPath'和'EmbedderPath'。

    返回:
        RAGLiteConfig: RAGLite的初始化配置对象。

    异常:
        ValueError: 如果配置过程中出现错误，例如设置字典中缺少键或值无效。"""
    try:
        return RAGLiteConfig(
            db_url=settings["DBUrl"],
            llm=f"llama-cpp-python/{settings['LLMPath']}",
            embedder=f"llama-cpp-python/{settings['EmbedderPath']}",
            embedder_normalize=True,
            chunk_max_size=512,
            reranker=Reranker("ms-marco-MiniLM-L-12-v2", model_type="flashrank")
        )
    except Exception as e:
        raise ValueError(f"配置错误: {e}")


def process_document(file_path: str) -> bool:
    """通过将文档插入具有给定配置的系统来处理文档。

    此函数尝试使用存储在会话状态中的预定义配置，将文件路径指定的文档插入到系统中。
    如果操作失败，它会记录错误。

    参数:
        file_path (str): 需要处理的文档文件的路径。

    返回:
        bool: 如果文档成功处理则为True；如果发生错误则为False。"""
    try:
        if not st.session_state.get('my_config'):
            raise ValueError("配置未初始化")
        insert_document(Path(file_path), config=st.session_state.my_config)
        return True
    except Exception as e:
        logger.error(f"处理文档时出错: {str(e)}")
        return False


def perform_search(query: str) -> List[dict]:
    """执行混合搜索并返回重新排序的结果。

    此函数使用提供的查询执行混合搜索，并尝试检索和重新排序相关块。
    它返回重新排序的搜索结果列表。

    参数:
        query (str): 搜索查询字符串。

    返回:
        List[dict]: 包含重新排序的搜索结果的字典列表。
        如果未找到结果或发生错误，则返回空列表。"""
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
        system_prompt = """你是一个有帮助的AI助手。当你不知道某些事情时，
        要诚实地说明。提供清晰、简洁和准确的回应。"""

        response_stream = rag(
            prompt=query,
            system_prompt=system_prompt,
            search=None,
            messages=[],
            max_tokens=1024,
            temperature=0.7,
            config=st.session_state.my_config
        )

        full_response = ""
        for chunk in response_stream:
            full_response += chunk

        if not full_response.strip():
            return "抱歉，我无法生成回应。请尝试重新表述你的问题。"

        return full_response

    except Exception as e:
        logger.error(f"备用处理错误: {str(e)}")
        return "抱歉，处理你的请求时遇到错误。请重试。"


def main():
    st.set_page_config(page_title="本地LLM驱动的混合搜索-RAG助手", layout="wide")

    for state_var in ['chat_history', 'documents_loaded', 'my_config']:
        if state_var not in st.session_state:
            st.session_state[
                state_var] = [] if state_var == 'chat_history' else False if state_var == 'documents_loaded' else None

    with st.sidebar:
        st.title("配置")

        llm_path = st.text_input(
            "LLM模型路径",
            value=st.session_state.get('llm_path', ''),
            placeholder="TheBloke/Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q4_K_M.gguf@4096",
            help="GGUF格式的本地LLM模型路径"
        )

        embedder_path = st.text_input(
            "嵌入模型路径",
            value=st.session_state.get('embedder_path', ''),
            placeholder="lm-kit/bge-m3-gguf/bge-m3-Q4_K_M.gguf@1024",
            help="GGUF格式的本地嵌入模型路径"
        )

        db_url = st.text_input(
            "数据库URL",
            value=st.session_state.get('db_url', ''),
            placeholder="postgresql://user:pass@host:port/db",
            help="数据库连接URL"
        )

        if st.button("保存配置"):
            try:
                if not all([llm_path, embedder_path, db_url]):
                    st.error("所有字段都是必填的！")
                    return

                settings = {
                    "LLMPath": llm_path,
                    "EmbedderPath": embedder_path,
                    "DBUrl": db_url
                }

                st.session_state.my_config = initialize_config(settings)
                st.success("配置保存成功！")

            except Exception as e:
                st.error(f"配置错误: {str(e)}")

    st.title("🖥️ 带有混合搜索的本地RAG应用")

    if st.session_state.my_config:
        uploaded_files = st.file_uploader(
            "上传PDF文档",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_uploader"
        )

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
                        logger.info("未找到相关文档。切换到本地LLM。")
                        with st.spinner("使用常识回答..."):
                            full_response = handle_fallback(user_input)
                            if full_response.startswith("抱歉"):
                                st.warning("未找到相关文档且备用回答失败。")
                            else:
                                st.info("基于常识回答。")
                    else:
                        formatted_messages = [
                            {"role": "user" if i % 2 == 0 else "assistant", "content": msg}
                            for i, msg in enumerate([m for pair in st.session_state.chat_history for m in pair])
                            if msg
                        ]

                        response_stream = rag(
                            prompt=user_input,
                            system_prompt=RAG_SYSTEM_PROMPT,
                            search=hybrid_search,
                            messages=formatted_messages,
                            max_contexts=5,
                            config=st.session_state.my_config
                        )

                        full_response = ""
                        for chunk in response_stream:
                            full_response += chunk
                            message_placeholder.markdown(full_response + "▌")

                    message_placeholder.markdown(full_response)
                    st.session_state.chat_history.append((user_input, full_response))

                except Exception as e:
                    logger.error(f"错误: {str(e)}")
                    st.error(f"错误: {str(e)}")
    else:
        st.info(
            "请配置你的模型路径并上传文档以开始使用。"
            if not st.session_state.my_config
            else "请上传一些文档以开始使用。"
        )


if __name__ == "__main__":
    main()