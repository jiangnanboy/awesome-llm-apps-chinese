import os
from uuid import uuid4
import requests
from agno.agent import Agent, RunResponse
from agno.models.openai import OpenAIChat
from agno.tools.models_labs import FileType, ModelsLabTools
from agno.utils.log import logger
import streamlit as st

# 侧边栏：用户输入API密钥
st.sidebar.title("API密钥配置")

openai_api_key = st.sidebar.text_input("输入您的OpenAI API密钥", type="password")
models_lab_api_key = st.sidebar.text_input("输入您的ModelsLab API密钥", type="password")

# Streamlit应用界面
st.title("🎶 ModelsLab音乐生成器")
prompt = st.text_area("输入音乐生成提示：", "生成一段30秒的古典音乐", height=100)

# 仅当提供了两个API密钥时才初始化代理
if openai_api_key and models_lab_api_key:
    agent = Agent(
        name="ModelsLab音乐代理",
        agent_id="ml_music_agent",
        model=OpenAIChat(id="gpt-4o", api_key=openai_api_key),
        show_tool_calls=True,
        tools=[ModelsLabTools(api_key=models_lab_api_key, wait_for_completion=True, file_type=FileType.MP3)],
        description="您是一个可以使用ModelsLabs API生成音乐的AI代理。",
        instructions=[
            "生成音乐时，使用`generate_media`工具并提供详细提示，指定：",
            "- 音乐的流派和风格（例如：古典、爵士、电子）",
            "- 要包含的乐器和声音",
            "- 速度、情绪和情感特质",
            "- 结构（前奏、主歌、副歌、桥段等）",
            "创建丰富、描述性的提示，以捕捉所需的音乐元素。",
            "专注于生成高质量、完整的器乐曲。",
        ],
        markdown=True,
        debug_mode=True,
    )

    if st.button("生成音乐"):
        if prompt.strip() == "":
            st.warning("请先输入提示内容。")
        else:
            with st.spinner("正在生成音乐... 🎵"):
                try:
                    music: RunResponse = agent.run(prompt)

                    if music.audio and len(music.audio) > 0:
                        save_dir = "audio_generations"
                        os.makedirs(save_dir, exist_ok=True)

                        url = music.audio[0].url
                        response = requests.get(url)

                        # 🛡️ 验证响应
                        if not response.ok:
                            st.error(f"下载音频失败。状态码：{response.status_code}")
                            st.stop()

                        content_type = response.headers.get("Content-Type", "")
                        if "audio" not in content_type:
                            st.error(f"返回的文件类型无效：{content_type}")
                            st.write("🔍 调试：下载的内容不是音频文件。")
                            st.write("🔗 URL：", url)
                            st.stop()

                        # ✅ 保存音频
                        filename = f"{save_dir}/music_{uuid4()}.mp3"
                        with open(filename, "wb") as f:
                            f.write(response.content)

                        # 🎧 播放音频
                        st.success("音乐生成成功！🎶")
                        audio_bytes = open(filename, "rb").read()
                        st.audio(audio_bytes, format="audio/mp3")

                        st.download_button(
                            label="下载音乐",
                            data=audio_bytes,
                            file_name="generated_music.mp3",
                            mime="audio/mp3"
                        )
                    else:
                        st.error("未生成音频。请重试。")

                except Exception as e:
                    st.error(f"发生错误：{e}")
                    logger.error(f"Streamlit应用错误：{e}")

else:
    st.sidebar.warning("请输入OpenAI和ModelsLab的API密钥以使用该应用。")