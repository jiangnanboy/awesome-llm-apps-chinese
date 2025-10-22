import os
from uuid import uuid4
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.eleven_labs import ElevenLabsTools
from agno.tools.firecrawl import FirecrawlTools
from agno.agent import Agent, RunResponse  # 注：此处重复导入Agent类，可能为代码冗余
from agno.utils.audio import write_audio_to_file
from agno.utils.log import logger
import streamlit as st

# Streamlit页面设置
st.set_page_config(page_title="📰 ➡️ 🎙️ 博客转播客工具", page_icon="🎙️")
st.title("📰 ➡️ 🎙️ 博客转播客工具")

# 侧边栏：API密钥
st.sidebar.header("🔑 API密钥")

openai_api_key = st.sidebar.text_input("OpenAI API密钥", type="password")  # 密码类型输入框，输入内容会隐藏
elevenlabs_api_key = st.sidebar.text_input("ElevenLabs API密钥", type="password")
firecrawl_api_key = st.sidebar.text_input("Firecrawl API密钥", type="password")

# 检查是否已提供所有密钥
keys_provided = all([openai_api_key, elevenlabs_api_key, firecrawl_api_key])  # 当三个密钥均非空时返回True

# 输入项：博客URL
url = st.text_input("输入博客URL：", "")  # 文本输入框，默认值为空字符串

# 按钮：生成播客
generate_button = st.button("🎙️ 生成播客", disabled=not keys_provided)  # 未提供全部密钥时按钮禁用

if not keys_provided:
    st.warning("请输入所有必需的API密钥，以启用播客生成功能。")  # 未提供全部密钥时显示警告

if generate_button:
    if url.strip() == "":  # 检查URL是否为空（去除首尾空格后）
        st.warning("请先输入博客URL。")
    else:
        # 将API密钥设置为环境变量，供Agno框架及工具使用
        os.environ["OPENAI_API_KEY"] = openai_api_key
        os.environ["ELEVEN_LABS_API_KEY"] = elevenlabs_api_key
        os.environ["FIRECRAWL_API_KEY"] = firecrawl_api_key

        # 显示加载中状态，提示用户当前进度
        with st.spinner("处理中... 正在抓取博客内容、生成摘要并制作播客 🎶"):
            try:
                # 初始化"博客转播客"智能体
                blog_to_podcast_agent = Agent(
                    name="博客转播客智能体",
                    agent_id="blog_to_podcast_agent",  # 智能体唯一标识
                    model=OpenAIChat(id="gpt-4o"),  # 使用OpenAI的gpt-4o模型
                    tools=[  # 为智能体配置所需工具
                        ElevenLabsTools(  # ElevenLabs工具：用于文本转语音
                            voice_id="JBFqnCBsd6RMkjVDRZzb",  # 语音ID（对应特定发音人）
                            model_id="eleven_multilingual_v2",  # 语音模型ID（多语言版本）
                            target_directory="audio_generations",  # 音频文件保存目录
                        ),
                        FirecrawlTools(),  # Firecrawl工具：用于抓取网页内容
                    ],
                    description="你是一个可通过ElevenLabs API生成音频的AI智能体。",
                    instructions=[  # 智能体执行步骤指令
                        "当用户提供博客URL时：",
                        "1. 使用FirecrawlTools抓取博客内容",
                        "2. 生成博客内容的简洁摘要，长度**不超过2000个字符**",
                        "3. 摘要需涵盖核心要点，同时保持生动性和口语化（符合播客风格）",
                        "4. 使用ElevenLabsTools将摘要转换为音频",
                        "注意：必须确保摘要在2000字符限制内，避免触发ElevenLabs API的长度限制"
                    ],
                    markdown=True,  # 支持Markdown格式输出
                    debug_mode=True,  # 启用调试模式（便于排查问题）
                )

                # 运行智能体，传入"将博客转为播客"的任务指令
                podcast: RunResponse = blog_to_podcast_agent.run(
                    f"将以下博客内容转换为播客：{url}"
                )

                # 定义音频保存目录，若目录不存在则创建
                save_dir = "audio_generations"
                os.makedirs(save_dir, exist_ok=True)  # exist_ok=True：目录存在时不报错

                # 检查是否成功生成音频（audio字段非空且长度大于0）
                if podcast.audio and len(podcast.audio) > 0:
                    # 生成唯一的音频文件名（使用uuid4避免文件名重复）
                    filename = f"{save_dir}/podcast_{uuid4()}.wav"
                    # 将base64编码的音频数据写入文件
                    write_audio_to_file(
                        audio=podcast.audio[0].base64_audio,  # 取第一个音频结果的base64数据
                        filename=filename
                    )

                    # 显示成功提示
                    st.success("播客生成成功！🎧")
                    # 读取音频文件并在页面中展示音频播放器
                    audio_bytes = open(filename, "rb").read()
                    st.audio(audio_bytes, format="audio/wav")

                    # 添加播客下载按钮
                    st.download_button(
                        label="下载播客",
                        data=audio_bytes,
                        file_name="generated_podcast.wav",  # 下载文件的默认名称
                        mime="audio/wav"  # 文件MIME类型（指定为wav格式）
                    )
                else:
                    # 未生成音频时显示错误提示
                    st.error("未生成任何音频文件，请重试。")

            # 捕获执行过程中的所有异常并提示
            except Exception as e:
                st.error(f"发生错误：{e}")  # 在页面显示错误信息
                logger.error(f"Streamlit应用错误：{e}")  # 将错误日志写入日志文件