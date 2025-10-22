import streamlit as st
import requests
from anthropic import Anthropic
import time
from typing import List, Dict, Optional
from urllib.parse import urlparse


class RAGPipeline:
    def __init__(self, ragie_api_key: str, anthropic_api_key: str):
        """
        使用API密钥初始化RAG流水线。
        """
        self.ragie_api_key = ragie_api_key
        self.anthropic_api_key = anthropic_api_key
        self.anthropic_client = Anthropic(api_key=anthropic_api_key)

        # API端点
        self.RAGIE_UPLOAD_URL = "https://api.ragie.ai/documents/url"
        self.RAGIE_RETRIEVAL_URL = "https://api.ragie.ai/retrievals"

    def upload_document(self, url: str, name: Optional[str] = None, mode: str = "fast") -> Dict:
        """
        从URL上传文档到Ragie。
        """
        if not name:
            # 从URL解析文档名称
            name = urlparse(url).path.split('/')[-1] or "document"

        payload = {
            "mode": mode,
            "name": name,
            "url": url
        }

        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {self.ragie_api_key}"
        }

        response = requests.post(self.RAGIE_UPLOAD_URL, json=payload, headers=headers)

        if not response.ok:
            raise Exception(f"文档上传失败: {response.status_code} {response.reason}")

        return response.json()

    def retrieve_chunks(self, query: str, scope: str = "tutorial") -> List[str]:
        """
        从Ragie检索与给定查询相关的文本片段。
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.ragie_api_key}"
        }

        payload = {
            "query": query,
            "filters": {
                "scope": scope
            }
        }

        response = requests.post(
            self.RAGIE_RETRIEVAL_URL,
            headers=headers,
            json=payload
        )

        if not response.ok:
            raise Exception(f"检索失败: {response.status_code} {response.reason}")

        data = response.json()
        return [chunk["text"] for chunk in data["scored_chunks"]]

    def create_system_prompt(self, chunk_texts: List[str]) -> str:
        """
        使用检索到的文本片段创建系统提示词。
        """
        return f"""请严格遵循以下指示：你是"Ragie AI"，一个专业且友好的AI聊天机器人，作为用户的助手。你当前的任务是根据下面提供的所有信息帮助用户。回答要非正式、直接且简洁，无需标题或问候语，但要包含所有相关内容。适当时使用富文本Markdown，包括加粗、斜体、段落和列表。如果使用LaTeX，请使用双$$作为分隔符，而不是单$。使用$$...$$代替括号。适当时将信息组织成多个部分或要点。不要包含来源中的原始项目ID或其他原始字段。除非用户要求，否则不要使用XML或其他标记。以下是可用于回答用户的所有信息：=== {chunk_texts} === 如果用户要求搜索但没有结果，请务必告知用户你找不到任何内容，以及他们可以做些什么来找到所需信息。结束系统指示"""

    def generate_response(self, system_prompt: str, query: str) -> str:
        """
        使用Claude 3.5 Sonnet生成回答。
        """
        message = self.anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1024,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": query
                }
            ]
        )

        return message.content[0].text

    def process_query(self, query: str, scope: str = "tutorial") -> str:
        """
        通过完整的RAG流水线处理查询。
        """
        chunks = self.retrieve_chunks(query, scope)

        if not chunks:
            return "未找到与您的查询相关的信息。"

        system_prompt = self.create_system_prompt(chunks)
        return self.generate_response(system_prompt, query)


def initialize_session_state():
    """初始化会话状态变量。"""
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    if 'document_uploaded' not in st.session_state:
        st.session_state.document_uploaded = False
    if 'api_keys_submitted' not in st.session_state:
        st.session_state.api_keys_submitted = False


def main():
    st.set_page_config(page_title="RAG即服务", layout="wide")
    initialize_session_state()

    st.title(":linked_paperclips: RAG即服务")

    # API密钥配置部分
    with st.expander("🔑 API密钥配置", expanded=not st.session_state.api_keys_submitted):
        col1, col2 = st.columns(2)
        with col1:
            ragie_key = st.text_input("Ragie API密钥", type="password", key="ragie_key")
        with col2:
            anthropic_key = st.text_input("Anthropic API密钥", type="password", key="anthropic_key")

        if st.button("提交API密钥"):
            if ragie_key and anthropic_key:
                try:
                    st.session_state.pipeline = RAGPipeline(ragie_key, anthropic_key)
                    st.session_state.api_keys_submitted = True
                    st.success("API密钥配置成功！")
                except Exception as e:
                    st.error(f"配置API密钥时出错：{str(e)}")
            else:
                st.error("请提供两个API密钥。")

    # 文档上传部分
    if st.session_state.api_keys_submitted:
        st.markdown("### 📄 文档上传")
        doc_url = st.text_input("输入文档URL")
        doc_name = st.text_input("文档名称（可选）")

        col1, col2 = st.columns([1, 3])
        with col1:
            upload_mode = st.selectbox("上传模式", ["fast（快速）", "accurate（精确）"])

        if st.button("上传文档"):
            if doc_url:
                try:
                    with st.spinner("正在上传文档..."):
                        st.session_state.pipeline.upload_document(
                            url=doc_url,
                            name=doc_name if doc_name else None,
                            mode=upload_mode.split("（")[0]  # 提取模式关键词
                        )
                        time.sleep(5)  # 等待索引完成
                        st.session_state.document_uploaded = True
                        st.success("文档上传并索引成功！")
                except Exception as e:
                    st.error(f"上传文档时出错：{str(e)}")
            else:
                st.error("请提供文档URL。")

    # 查询部分
    if st.session_state.document_uploaded:
        st.markdown("### 🔍 文档查询")
        query = st.text_input("输入您的查询")

        if st.button("生成回答"):
            if query:
                try:
                    with st.spinner("正在生成回答..."):
                        response = st.session_state.pipeline.process_query(query)
                        st.markdown("### 回答：")
                        st.markdown(response)
                except Exception as e:
                    st.error(f"生成回答时出错：{str(e)}")
            else:
                st.error("请输入查询内容。")


if __name__ == "__main__":
    main()