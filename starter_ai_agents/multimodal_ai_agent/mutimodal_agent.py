import streamlit as st
from agno.agent import Agent
from agno.models.google import Gemini
from agno.media import Video
import time
from pathlib import Path
import tempfile

st.set_page_config(
    page_title="多模态AI代理",
    page_icon="🧬",
    layout="wide"
)

st.title("多模态AI代理 🧬")

# 从用户获取Gemini API密钥
gemini_api_key = st.text_input("输入您的Gemini API密钥", type="password")


# 初始化具备两种功能的单一代理
@st.cache_resource
def initialize_agent(api_key):
    return Agent(
        name="多模态分析师",
        model=Gemini(id="gemini-2.0-flash", api_key=api_key),
        markdown=True,
    )


if gemini_api_key:
    agent = initialize_agent(gemini_api_key)

    # 文件上传器
    uploaded_file = st.file_uploader("上传视频文件", type=['mp4', 'mov', 'avi'])

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name

        st.video(video_path)

        user_prompt = st.text_area(
            "您想了解什么？",
            placeholder="提出任何与视频相关的问题 - AI代理将分析视频并在需要时搜索网络",
            help="您可以询问有关视频内容的问题，并从网络获取相关信息"
        )

        if st.button("分析与研究"):
            if not user_prompt:
                st.warning("请输入您的问题。")
            else:
                try:
                    with st.spinner("正在处理视频并进行研究..."):
                        video = Video(filepath=video_path)

                        prompt = f"""
                        首先分析此视频，然后使用视频分析和网络研究来回答以下问题：{user_prompt}

                        提供全面的响应，重点关注实用、可操作的信息。
                        """

                        result = agent.run(prompt, videos=[video])

                    st.subheader("结果")
                    st.markdown(result.content)

                except Exception as e:
                    st.error(f"发生错误：{str(e)}")
                finally:
                    Path(video_path).unlink(missing_ok=True)
    else:
        st.info("请上传视频以开始分析。")
else:
    st.warning("请输入您的Gemini API密钥以继续。")

st.markdown("""
    <style>
    .stTextArea textarea {
        height: 100px;
    }
    </style>
    """, unsafe_allow_html=True)