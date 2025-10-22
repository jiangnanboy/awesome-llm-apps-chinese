from agno.agent import Agent
from agno.models.google import Gemini
from agno.media import Image as AgnoImage  # 导入AgnoImage类并指定别名，避免与其他Image类冲突
from agno.tools.duckduckgo import DuckDuckGoTools  # 导入DuckDuckGo搜索工具
import streamlit as st  # 导入streamlit库，用于构建Web界面，别名设为st
from typing import List, Optional  # 导入类型提示工具，用于指定函数参数和返回值类型
import logging  # 导入日志模块，用于记录程序运行中的错误信息
from pathlib import Path  # 导入Path类，用于处理文件路径
import tempfile  # 导入临时文件模块，用于创建临时文件存储上传的图片
import os  # 导入操作系统模块，用于处理文件系统相关操作

# 配置日志记录，仅记录错误级别（ERROR）及以上的日志
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器实例


def initialize_agents(api_key: str) -> tuple[Agent, Agent, Agent, Agent]:
    """
    初始化四个AI智能体（Agent）：心理咨询智能体、情感了结智能体、日程规划智能体、坦诚反馈智能体

    参数：
        api_key: Google Gemini API的密钥，用于调用Gemini模型

    返回：
        包含四个Agent实例的元组；若初始化失败，返回四个None组成的元组
    """
    try:
        # 初始化Gemini模型，指定模型ID为"gemini-2.0-flash-exp"（Gemini 2.0闪电实验版）
        model = Gemini(id="gemini-2.0-flash-exp", api_key=api_key)

        # 1. 心理咨询智能体（Therapist Agent）：提供共情支持和情感疏导
        therapist_agent = Agent(
            model=model,  # 绑定Gemini模型
            name="Therapist Agent",  # 智能体名称
            instructions=[  # 智能体的核心指令
                "你是一位富有同理心的心理咨询师，需做到：",
                "1. 带着同理心倾听，并认可用户的情绪感受",
                "2. 用温和的幽默缓解沉重氛围",
                "3. 分享有共鸣的分手经历",
                "4. 提供安慰性的话语和鼓励",
                "5. 分析文本和图片输入中的情感背景信息",
                "在回复中保持支持性和理解的态度"
            ],
            markdown=True  # 允许回复使用Markdown格式排版
        )

        # 2. 情感了结智能体（Closure Agent）：帮助用户处理未表达的情感，实现情感了结
        closure_agent = Agent(
            model=model,
            name="Closure Agent",
            instructions=[
                "你是一位情感了结专家，需做到：",
                "1. 为未表达的情感创建情绪化信息（如未发送的消息）",
                "2. 帮助用户表达真实、直白的情绪",
                "3. 用标题清晰地格式化信息内容",
                "4. 确保语气真挚且真实",
                "核心聚焦于情感释放和情感了结"
            ],
            markdown=True
        )

        # 3. 恢复日程规划智能体（Routine Planner Agent）：设计实用的分手恢复期日程
        routine_planner_agent = Agent(
            model=model,
            name="Routine Planner Agent",
            instructions=[
                "你是一位恢复期日程规划师，需做到：",
                "1. 设计7天恢复期挑战计划",
                "2. 包含有趣的活动和自我关怀任务",
                "3. 提出社交媒体戒断策略",
                "4. 创建能增强信心的播放列表",
                "核心聚焦于实用的恢复步骤"
            ],
            markdown=True
        )

        # 4. 坦诚反馈智能体（Brutal Honesty Agent）：提供直接、客观的分手分析
        brutal_honesty_agent = Agent(
            model=model,
            name="Brutal Honesty Agent",
            tools=[DuckDuckGoTools()],  # 绑定DuckDuckGo搜索工具，可用于获取外部信息
            instructions=[
                "你是一位直接反馈专家，需做到：",
                "1. 针对分手提供直白、客观的反馈",
                "2. 清晰解释关系失败的原因",
                "3. 使用坦率、基于事实的语言",
                "4. 提供向前看的理由",
                "核心聚焦于真实见解，不粉饰太平"
            ],
            markdown=True
        )

        # 返回四个初始化成功的智能体
        return therapist_agent, closure_agent, routine_planner_agent, brutal_honesty_agent

    # 捕获初始化过程中可能出现的所有异常（如API密钥无效、网络错误等）
    except Exception as e:
        st.error(f"智能体初始化失败：{str(e)}")  # 在Web界面显示错误信息
        return None, None, None, None  # 返回四个None表示初始化失败


# 配置Streamlit页面基础设置
st.set_page_config(
    page_title="💔 分手恢复期支持团队",  # 页面标题（显示在浏览器标签栏）
    page_icon="💔",  # 页面图标（显示在浏览器标签栏）
    layout="wide"  # 页面布局：宽屏模式
)

# 侧边栏：用于输入API密钥
with st.sidebar:
    st.header("🔑 API配置")  # 侧边栏标题

    # 初始化会话状态（session_state）：存储API密钥输入，避免页面刷新后丢失
    if "api_key_input" not in st.session_state:
        st.session_state.api_key_input = ""  # 初始值为空字符串

    # 创建API密钥输入框（密码类型，输入内容会隐藏）
    api_key = st.text_input(
        "输入你的Gemini API密钥",
        value=st.session_state.api_key_input,  # 初始值为会话状态中存储的密钥
        type="password",  # 密码类型，输入内容显示为星号
        help="从Google AI Studio获取你的API密钥",  # 输入框提示信息（鼠标悬浮时显示）
        key="api_key_widget"  # 组件唯一标识，用于Streamlit跟踪组件状态
    )

    # 若输入的密钥与会话状态中存储的不同，更新会话状态
    if api_key != st.session_state.api_key_input:
        st.session_state.api_key_input = api_key

    # 根据是否输入API密钥显示不同提示
    if api_key:
        st.success("已提供API密钥！✅")  # 成功提示（绿色）
    else:
        st.warning("请输入API密钥以继续")  # 警告提示（黄色）
        # 显示获取API密钥的步骤（Markdown格式）
        st.markdown("""
        获取API密钥的步骤：
        1. 访问 [Google AI Studio](https://makersuite.google.com/app/apikey)
        2. 在你的 [Google Cloud控制台](https://console.developers.google.com/apis/api/generativelanguage.googleapis.com) 中启用「生成式语言API」（Generative Language API）
        """)

# 主内容区
st.title("💔 分手恢复期支持团队")  # 主标题
# 主内容说明（Markdown格式）
st.markdown("""
    ### 你的AI驱动分手恢复期支持团队已就位！
    分享你的感受和聊天截图，我们会帮你度过这段艰难的时光。
""")

# 输入区域：分为两列（情感分享列 + 截图上传列）
col1, col2 = st.columns(2)  # 创建两列布局

# 第一列：情感分享（用户输入自身感受）
with col1:
    st.subheader("分享你的感受")  # 列标题
    # 创建多行文本输入框（用于用户输入感受或经历）
    user_input = st.text_area(
        "你现在感觉如何？发生了什么事？",
        height=150,  # 输入框高度（像素）
        placeholder="告诉我们你的故事..."  # 输入框提示文本（未输入时显示）
    )

# 第二列：聊天截图上传（可选，用于提供更多情感背景）
with col2:
    st.subheader("上传聊天截图")  # 列标题
    # 创建文件上传器（支持多文件上传，仅允许jpg/jpeg/png格式）
    uploaded_files = st.file_uploader(
        "上传你的聊天截图（可选）",
        type=["jpg", "jpeg", "png"],  # 允许上传的文件类型
        accept_multiple_files=True,  # 允许上传多个文件
        key="screenshots"  # 组件唯一标识
    )

    # 若有上传的文件，在界面上显示每个图片（带文件名标题）
    if uploaded_files:
        for file in uploaded_files:
            st.image(file, caption=file.name, use_container_width=True)  # 图片宽度适应列宽

# 处理按钮与API密钥检查：点击按钮后生成恢复计划
if st.button("获取恢复计划 💝", type="primary"):  # 创建primary类型按钮（主要操作按钮，颜色较深）
    # 检查会话状态中是否存储了API密钥：若无，显示警告
    if not st.session_state.api_key_input:
        st.warning("请先在侧边栏输入你的API密钥！")
    else:
        # 调用initialize_agents函数，传入API密钥初始化四个智能体
        therapist_agent, closure_agent, routine_planner_agent, brutal_honesty_agent = initialize_agents(
            st.session_state.api_key_input)

        # 检查所有智能体是否都初始化成功
        if all([therapist_agent, closure_agent, routine_planner_agent, brutal_honesty_agent]):
            # 检查用户是否提供了输入（感受或截图）：至少需一项
            if user_input or uploaded_files:
                try:
                    st.header("你的个性化恢复计划")  # 生成恢复计划的标题


                    # 定义图片处理函数：将上传的Streamlit文件对象转换为AgnoImage对象
                    def process_images(files):
                        processed_images = []  # 存储处理后的AgnoImage对象
                        for file in files:
                            try:
                                # 获取系统临时目录路径
                                temp_dir = tempfile.gettempdir()
                                # 创建临时文件路径（避免文件名冲突，前缀加"temp_"）
                                temp_path = os.path.join(temp_dir, f"temp_{file.name}")

                                # 将上传的文件内容写入临时文件
                                with open(temp_path, "wb") as f:
                                    f.write(file.getvalue())

                                # 基于临时文件路径创建AgnoImage对象（供智能体处理图片）
                                agno_image = AgnoImage(filepath=Path(temp_path))
                                processed_images.append(agno_image)

                            # 捕获单张图片处理失败的异常（不中断整体流程，仅记录日志）
                            except Exception as e:
                                logger.error(f"处理图片 {file.name} 时出错：{str(e)}")
                                continue
                        return processed_images


                    # 处理上传的图片：若有上传文件则调用process_images，否则为空列表
                    all_images = process_images(uploaded_files) if uploaded_files else []

                    # 1. 心理咨询智能体：生成情感支持内容
                    with st.spinner("🤗 正在获取共情支持..."):  # 显示加载中动画和提示文本
                        # 构建心理咨询智能体的提示词（包含用户输入和任务要求）
                        therapist_prompt = f"""
                        基于以下信息分析用户的情绪状态，并提供共情支持：
                        用户的留言：{user_input}

                        请提供富有同情心的回复，包含：
                        1. 对用户情绪的认可
                        2. 温和的安慰话语
                        3. 有共鸣的经历分享
                        4. 鼓励性的话语
                        """

                        # 调用心理咨询智能体，传入提示词和处理后的图片（若有）
                        response = therapist_agent.run(
                            message=therapist_prompt,
                            images=all_images
                        )

                        # 在界面上显示心理咨询结果
                        st.subheader("🤗 情感支持")
                        st.markdown(response.content)  # 显示智能体回复内容（Markdown格式）

                    # 2. 情感了结智能体：生成情感了结相关内容（如未发送消息模板）
                    with st.spinner("✍️ 正在撰写情感了结内容..."):
                        closure_prompt = f"""
                        基于以下信息帮助用户实现情感了结：
                        用户的感受：{user_input}

                        请提供：
                        1. 未发送消息的模板
                        2. 情感释放练习
                        3. 情感了结仪式建议
                        4. 向前看的策略
                        """

                        response = closure_agent.run(
                            message=closure_prompt,
                            images=all_images
                        )

                        st.subheader("✍️ 情感了结")
                        st.markdown(response.content)

                    # 3. 恢复日程规划智能体：生成7天恢复计划
                    with st.spinner("📅 正在创建你的恢复计划..."):
                        routine_prompt = f"""
                        基于以下信息设计7天恢复计划：
                        用户当前状态：{user_input}

                        计划需包含：
                        1. 每日活动和挑战
                        2. 自我关怀流程
                        3. 社交媒体使用指南
                        4. 改善心情的音乐建议
                        """

                        response = routine_planner_agent.run(
                            message=routine_prompt,
                            images=all_images
                        )

                        st.subheader("📅 你的恢复计划")
                        st.markdown(response.content)

                    # 4. 坦诚反馈智能体：生成客观、直接的分手分析
                    with st.spinner("💪 正在获取坦诚视角..."):
                        honesty_prompt = f"""
                        基于以下信息提供坦诚、有建设性的反馈：
                        情况描述：{user_input}

                        反馈需包含：
                        1. 客观分析
                        2. 成长机会
                        3. 未来展望
                        4. 可执行的步骤
                        """

                        response = brutal_honesty_agent.run(
                            message=honesty_prompt,
                            images=all_images
                        )

                        st.subheader("💪 坦诚视角")
                        st.markdown(response.content)

                # 捕获恢复计划生成过程中的所有异常（如智能体调用失败、图片处理错误等）
                except Exception as e:
                    logger.error(f"分析过程中出错：{str(e)}")  # 记录错误日志
                    st.error("分析过程中发生错误，请查看日志获取详细信息。")  # 在界面显示错误
            else:
                # 若用户未提供任何输入（感受或截图），显示警告
                st.warning("请分享你的感受或上传截图以获取帮助。")
        else:
            # 若智能体初始化失败，显示错误
            st.error("智能体初始化失败，请检查你的API密钥。")

# 页脚：显示版权信息和话题标签（使用HTML格式实现居中对齐）
st.markdown("---")  # 显示一条水平线（分隔线）
st.markdown("""
    <div style='text-align: center'>
        <p>由分手恢复期支持团队 ❤️ 开发</p>
        <p>用 #分手恢复期支持团队 分享你的恢复旅程</p>
    </div>
""", unsafe_allow_html=True)  # 允许使用HTML格式（需设置unsafe_allow_html=True）
