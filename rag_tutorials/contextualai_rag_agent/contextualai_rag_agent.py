import os
import tempfile
import time
from typing import List, Optional, Tuple, Any

import streamlit as st
import requests
import json
import re
from contextual import ContextualAI


def init_session_state() -> None:
    """初始化会话状态变量，确保所有需要的状态都已定义"""
    if "api_key_submitted" not in st.session_state:
        st.session_state.api_key_submitted = False
    if "contextual_api_key" not in st.session_state:
        st.session_state.contextual_api_key = ""
    if "base_url" not in st.session_state:
        st.session_state.base_url = "https://api.contextual.ai/v1"
    if "agent_id" not in st.session_state:
        st.session_state.agent_id = ""
    if "datastore_id" not in st.session_state:
        st.session_state.datastore_id = ""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processed_file" not in st.session_state:
        st.session_state.processed_file = False
    if "last_raw_response" not in st.session_state:
        st.session_state.last_raw_response = None
    if "last_user_query" not in st.session_state:
        st.session_state.last_user_query = ""


def sidebar_api_form() -> bool:
    """侧边栏API设置表单，用于输入和验证API密钥及资源ID"""
    with st.sidebar:
        st.header("API和资源设置")

        if st.session_state.api_key_submitted:
            st.success("API验证通过")
            if st.button("重置设置"):
                st.session_state.clear()
                st.rerun()
            return True

        with st.form("contextual_api_form"):
            api_key = st.text_input("Contextual AI API密钥", type="password")
            base_url = st.text_input(
                "基础URL",
                value=st.session_state.base_url,
                help="包含/v1（例如：https://api.contextual.ai/v1）",
            )
            existing_agent_id = st.text_input("已有Agent ID（可选）")
            existing_datastore_id = st.text_input("已有数据存储ID（可选）")

            if st.form_submit_button("保存并验证"):
                try:
                    client = ContextualAI(api_key=api_key, base_url=base_url)
                    _ = client.agents.list()  # 验证API密钥是否有效

                    # 保存配置到会话状态
                    st.session_state.contextual_api_key = api_key
                    st.session_state.base_url = base_url
                    st.session_state.agent_id = existing_agent_id
                    st.session_state.datastore_id = existing_datastore_id
                    st.session_state.api_key_submitted = True

                    st.success("凭据验证成功！")
                    st.rerun()
                except Exception as e:
                    st.error(f"凭据验证失败: {str(e)}")
        return False


def ensure_client():
    """确保客户端已正确初始化，若未提供API密钥则抛出错误"""
    if not st.session_state.get("contextual_api_key"):
        raise ValueError("未提供Contextual AI API密钥")
    return ContextualAI(api_key=st.session_state.contextual_api_key, base_url=st.session_state.base_url)


def create_datastore(client, name: str) -> Optional[str]:
    """创建新的数据存储并返回其ID"""
    try:
        ds = client.datastores.create(name=name)
        return getattr(ds, "id", None)
    except Exception as e:
        st.error(f"创建数据存储失败: {e}")
        return None


# 允许上传的文件扩展名
ALLOWED_EXTS = {".pdf", ".html", ".htm", ".mhtml", ".doc", ".docx", ".ppt", ".pptx"}


def upload_documents(client, datastore_id: str, files: List[bytes], filenames: List[str], metadata: Optional[dict]) -> \
List[str]:
    """将文档上传到指定的数据存储，并返回文档ID列表"""
    doc_ids: List[str] = []
    for content, fname in zip(files, filenames):
        try:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in ALLOWED_EXTS:
                st.error(f"{fname}的文件扩展名不支持。允许的格式: {sorted(ALLOWED_EXTS)}")
                continue

            # 创建临时文件处理上传内容
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            # 上传临时文件到数据存储
            with open(tmp_path, "rb") as f:
                if metadata:
                    result = client.datastores.documents.ingest(datastore_id, file=f, metadata=metadata)
                else:
                    result = client.datastores.documents.ingest(datastore_id, file=f)
                doc_ids.append(getattr(result, "id", ""))

        except Exception as e:
            st.error(f"上传{fname}失败: {e}")
        finally:
            # 确保临时文件被删除
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
    return doc_ids


def wait_until_documents_ready(api_key: str, datastore_id: str, base_url: str, max_checks: int = 30,
                               interval_sec: float = 5.0) -> None:
    """等待文档处理完成，定期检查文档状态"""
    url = f"{base_url.rstrip('/')}/datastores/{datastore_id}/documents"
    headers = {"Authorization": f"Bearer {api_key}"}

    for _ in range(max_checks):
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 200:
                docs = resp.json().get("documents", [])
                # 检查是否所有文档都已处理完毕
                if not any(d.get("status") in ("processing", "pending") for d in docs):
                    return
            time.sleep(interval_sec)
        except Exception:
            time.sleep(interval_sec)


def create_agent(client, name: str, description: str, datastore_id: str) -> Optional[str]:
    """创建新的智能体并关联到指定的数据存储，返回智能体ID"""
    try:
        agent = client.agents.create(name=name, description=description, datastore_ids=[datastore_id])
        return getattr(agent, "id", None)
    except Exception as e:
        st.error(f"创建智能体失败: {e}")
        return None


def query_agent(client, agent_id: str, query: str) -> Tuple[str, Any]:
    """向智能体发送查询并返回响应结果"""
    try:
        resp = client.agents.query.create(agent_id=agent_id, messages=[{"role": "user", "content": query}])

        # 处理不同格式的响应
        if hasattr(resp, "content"):
            return resp.content, resp
        if hasattr(resp, "message") and hasattr(resp.message, "content"):
            return resp.message.content, resp
        if hasattr(resp, "messages") and resp.messages:
            last_msg = resp.messages[-1]
            return getattr(last_msg, "content", str(last_msg)), resp
        return str(resp), resp
    except Exception as e:
        return f"查询智能体出错: {e}", None


def show_retrieval_info(client, raw_response, agent_id: str) -> None:
    """显示检索信息，包括相关文档的页面图像"""
    try:
        if not raw_response:
            st.info("没有可用的检索信息。")
            return

        message_id = getattr(raw_response, "message_id", None)
        retrieval_contents = getattr(raw_response, "retrieval_contents", [])

        if not message_id or not retrieval_contents:
            st.info("未返回检索元数据。")
            return

        first_content_id = getattr(retrieval_contents[0], "content_id", None)
        if not first_content_id:
            st.info("检索元数据中缺少content_id。")
            return

        # 获取检索内容的详细信息
        ret_result = client.agents.query.retrieval_info(
            message_id=message_id,
            agent_id=agent_id,
            content_ids=[first_content_id]
        )

        metadatas = getattr(ret_result, "content_metadatas", [])
        if not metadatas:
            st.info("未找到内容元数据。")
            return

        page_img_b64 = getattr(metadatas[0], "page_img", None)
        if not page_img_b64:
            st.info("元数据中没有提供页面图像。")
            return

        # 解码并显示图像
        import base64
        img_bytes = base64.b64decode(page_img_b64)
        st.image(img_bytes, caption="主要引用页面", use_container_width=True)

    except Exception as e:
        st.error(f"加载检索信息失败: {e}")


def update_agent_prompt(client, agent_id: str, system_prompt: str) -> bool:
    """更新智能体的系统提示词"""
    try:
        client.agents.update(agent_id=agent_id, system_prompt=system_prompt)
        return True
    except Exception as e:
        st.error(f"更新系统提示词失败: {e}")
        return False


def evaluate_with_lmunit(client, query: str, response_text: str, unit_test: str):
    """使用LMUnit评估智能体的回答质量"""
    try:
        result = client.lmunit.create(query=query, response=response_text, unit_test=unit_test)
        st.subheader("评估结果")
        st.code(str(result), language="json")
    except Exception as e:
        st.error(f"LMUnit评估失败: {e}")


def post_process_answer(text: str) -> str:
    """处理回答文本，优化格式"""
    text = re.sub(r"\(\s*\)", "", text)  # 移除空括号
    text = text.replace("• ", "\n- ")  # 将项目符号转换为更易读的格式
    return text


# 初始化会话状态
init_session_state()

# 设置页面标题
st.title("Contextual AI 检索增强生成智能体")

# 检查API设置，若未完成则显示提示
if not sidebar_api_form():
    st.info("请在侧边栏输入您的Contextual AI API密钥以继续。")
    st.stop()

# 确保客户端已初始化
client = ensure_client()

# 1) 创建或选择数据存储
with st.expander("1) 创建或选择数据存储", expanded=True):
    if not st.session_state.datastore_id:
        default_name = "contextualai_rag_datastore"
        ds_name = st.text_input("数据存储名称", value=default_name)
        if st.button("创建数据存储"):
            ds_id = create_datastore(client, ds_name)
            if ds_id:
                st.session_state.datastore_id = ds_id
                st.success(f"已创建数据存储: {ds_id}")
    else:
        st.success(f"正在使用数据存储: {st.session_state.datastore_id}")

# 2) 上传文档
with st.expander("2) 上传文档", expanded=True):
    uploaded_files = st.file_uploader(
        "上传PDF或文本文件",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True
    )
    metadata_json = st.text_area(
        "自定义元数据（JSON格式）",
        value="",
        placeholder='{"custom_metadata": {"field1": "value1"}}'
    )

    if uploaded_files and st.session_state.datastore_id:
        contents = [f.getvalue() for f in uploaded_files]
        names = [f.name for f in uploaded_files]

        if st.button("导入文档"):
            parsed_metadata = None
            if metadata_json.strip():
                try:
                    parsed_metadata = json.loads(metadata_json)
                except Exception as e:
                    st.error(f"无效的元数据JSON: {e}")
                    parsed_metadata = None

            # 上传文档
            ids = upload_documents(
                client,
                st.session_state.datastore_id,
                contents,
                names,
                parsed_metadata
            )

            if ids:
                st.success(f"已上传 {len(ids)} 个文档")
                # 等待文档处理完成
                wait_until_documents_ready(
                    st.session_state.contextual_api_key,
                    st.session_state.datastore_id,
                    st.session_state.base_url
                )
                st.info("文档已准备就绪。")

# 3) 创建或选择智能体
with st.expander("3) 创建或选择智能体", expanded=True):
    if not st.session_state.agent_id and st.session_state.datastore_id:
        agent_name = st.text_input("智能体名称", value="ContextualAI RAG智能体")
        agent_desc = st.text_area("智能体描述", value="基于上传文档的检索增强生成智能体")

        if st.button("创建智能体"):
            a_id = create_agent(client, agent_name, agent_desc, st.session_state.datastore_id)
            if a_id:
                st.session_state.agent_id = a_id
                st.success(f"已创建智能体: {a_id}")
    elif st.session_state.agent_id:
        st.success(f"正在使用智能体: {st.session_state.agent_id}")

# 4) 智能体设置（可选）
with st.expander("4) 智能体设置（可选）"):
    if st.session_state.agent_id:
        system_prompt_val = st.text_area(
            "系统提示词",
            value="",
            placeholder="粘贴新的系统提示词以更新您的智能体"
        )
        if st.button("更新系统提示词") and system_prompt_val.strip():
            ok = update_agent_prompt(client, st.session_state.agent_id, system_prompt_val.strip())
            if ok:
                st.success("系统提示词已更新。")

# 分割线
st.divider()

# 显示聊天历史
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

    # 处理用户查询
query = st.chat_input("询问关于您文档的问题")
if query:
    st.session_state.last_user_query = query
    st.session_state.chat_history.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    if st.session_state.agent_id:
        with st.chat_message("assistant"):
            answer, raw = query_agent(client, st.session_state.agent_id, query)
            st.session_state.last_raw_response = raw
            processed = post_process_answer(answer)
            st.markdown(processed)
            st.session_state.chat_history.append({"role": "assistant", "content": processed})
    else:
        st.error("请先创建或选择一个智能体。")

# 调试和评估部分
with st.expander("调试与评估", expanded=False):
    st.caption("用于检查检索内容和评估回答的工具")

    if st.session_state.agent_id:
        if st.checkbox("显示检索信息", value=False):
            show_retrieval_info(client, st.session_state.last_raw_response, st.session_state.agent_id)

        st.markdown("")
        unit_test = st.text_area(
            "LMUnit评分标准/单元测试",
            value="回答是否避免了不必要的信息？",
            height=80
        )

        if st.button("用LMUnit评估上一个回答"):
            if st.session_state.last_user_query and st.session_state.chat_history:
                # 获取上一个助手回答
                last_assistant_msgs = [m for m in st.session_state.chat_history if m["role"] == "assistant"]
                if last_assistant_msgs:
                    evaluate_with_lmunit(
                        client,
                        st.session_state.last_user_query,
                        last_assistant_msgs[-1]["content"],
                        unit_test
                    )
                else:
                    st.info("还没有可评估的助手回答。")
            else:
                st.info("请先提问以运行评估。")

# 侧边栏底部的控制按钮
with st.sidebar:
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("清除聊天"):
            st.session_state.chat_history = []
            st.session_state.last_raw_response = None
            st.session_state.last_user_query = ""
            st.rerun()
    with col2:
        if st.button("重置应用"):
            st.session_state.clear()
            st.rerun()