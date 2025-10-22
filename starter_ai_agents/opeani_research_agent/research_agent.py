import os
import uuid
import asyncio
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv

from agents import (
    Agent,
    Runner,
    WebSearchTool,
    function_tool,
    handoff,
    trace,
)

from pydantic import BaseModel

# 加载环境变量
load_dotenv()

# 设置页面配置
st.set_page_config(
    page_title="OpenAI 研究助手智能体",
    page_icon="📰",
    layout="wide",  # 宽屏布局
    initial_sidebar_state="expanded"  # 侧边栏默认展开
)

# 确保API密钥已配置
if not os.environ.get("OPENAI_API_KEY"):
    st.error("请设置您的 OPENAI_API_KEY 环境变量")
    st.stop()  # 停止页面运行

# 应用标题与描述
st.title("📰 OpenAI 研究助手智能体")
st.subheader("由 OpenAI Agents SDK 提供支持")
st.markdown("""
本应用通过创建多智能体系统展示 OpenAI Agents SDK 的能力，
该系统可研究新闻主题并生成全面的研究报告。
""")


# 定义数据模型
class ResearchPlan(BaseModel):
    topic: str  # 研究主题
    search_queries: list[str]  # 搜索查询列表
    focus_areas: list[str]  # 重点研究领域列表


class ResearchReport(BaseModel):
    title: str  # 报告标题
    outline: list[str]  # 报告大纲
    report: str  # 报告正文
    sources: list[str]  # 信息来源列表
    word_count: int  # 报告字数


# 用于保存研究过程中发现的事实的自定义工具
@function_tool
def save_important_fact(fact: str, source: str = None) -> str:
    """保存研究过程中发现的重要事实。

    参数:
        fact: 需保存的重要事实
        source: 事实的可选来源

    返回:
        确认信息
    """
    if "collected_facts" not in st.session_state:
        st.session_state.collected_facts = []  # 初始化事实存储列表

    st.session_state.collected_facts.append({
        "fact": fact,
        "source": source or "未指定",  # 若未提供来源则显示"未指定"
        "timestamp": datetime.now().strftime("%H:%M:%S")  # 记录事实保存时间
    })

    return f"事实已保存: {fact}"


# 定义智能体
research_agent = Agent(
    name="研究智能体",
    instructions="你是一名研究助手。给定搜索关键词后，需通过网络搜索该关键词并"
                 "生成结果的简洁摘要。摘要需包含2-3个段落，字数控制在300字以内。"
                 "需捕捉核心要点，语言简洁（无需完整句子或规范语法）。"
                 "该摘要将供报告整合人员使用，因此务必提炼精华、剔除无关内容。"
                 "除摘要本身外，不得包含任何额外评论。",
    model="gpt-4o-mini",  # 使用的模型
    tools=[
        WebSearchTool(),  # 网络搜索工具
        save_important_fact  # 事实保存工具
    ],
)

editor_agent = Agent(
    name="编辑智能体",
    handoff_description="负责撰写全面研究报告的高级研究员",
    instructions="你是一名高级研究员，负责为研究需求撰写结构完整的报告。"
                 "你将收到原始研究需求及研究助手完成的初步研究内容。\n"
                 "首先需为报告制定大纲，明确报告的结构与逻辑流程。"
                 "然后生成完整报告作为最终输出。\n"
                 "最终输出需采用Markdown格式，内容需详实深入，目标长度为5-10页，"
                 "字数至少1000字。",
    model="gpt-4o-mini",
    output_type=ResearchReport,  # 输出类型为ResearchReport模型
)

triage_agent = Agent(
    name="调度智能体",
    instructions="""你是本次研究任务的协调者。主要职责包括：
    1. 理解用户提出的研究主题
    2. 制定研究计划，包含以下要素：
       - topic: 清晰的研究主题表述
       - search_queries: 3-5个用于收集信息的具体搜索查询
       - focus_areas: 3-5个需深入研究的主题关键方向
    3. 将任务移交至研究智能体以收集信息
    4. 研究完成后，将任务移交至编辑智能体撰写全面报告

    请确保研究计划以指定格式呈现，包含topic、search_queries和focus_areas三个部分。
    """,
    handoffs=[
        handoff(research_agent),  # 移交至研究智能体
        handoff(editor_agent)  # 移交至编辑智能体
    ],
    model="gpt-4o-mini",
    output_type=ResearchPlan,  # 输出类型为ResearchPlan模型
)

# 创建用于输入和控制的侧边栏
with st.sidebar:
    st.header("研究主题")
    user_topic = st.text_input(
        "输入需研究的主题：",
    )

    # 开始研究按钮（无主题时禁用）
    start_button = st.button("开始研究", type="primary", disabled=not user_topic)

    st.divider()  # 分隔线
    st.subheader("示例主题")
    example_topics = [
        "对于从未乘坐过邮轮的首次旅行者，美国最佳邮轮公司有哪些？",
        "对于想从法式压滤壶升级的用户，最佳高性价比意式浓缩咖啡机有哪些？",
        "对于首次独自旅行的人，印度最佳小众旅行目的地有哪些？"
    ]

    # 示例主题按钮（点击后自动填充至输入框）
    for topic in example_topics:
        if st.button(topic):
            user_topic = topic
            start_button = True

# 主内容区（包含两个标签页）
tab1, tab2 = st.tabs(["研究过程", "报告"])

# 初始化会话状态以存储结果
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4().hex[:16])  # 生成唯一会话ID
if "collected_facts" not in st.session_state:
    st.session_state.collected_facts = []  # 存储收集到的事实
if "research_done" not in st.session_state:
    st.session_state.research_done = False  # 研究完成状态标记
if "report_result" not in st.session_state:
    st.session_state.report_result = None  # 存储报告结果


# 主研究函数（异步）
async def run_research(topic):
    # 重置新研究的状态
    st.session_state.collected_facts = []
    st.session_state.research_done = False
    st.session_state.report_result = None

    with tab1:
        message_container = st.container()  # 用于显示研究过程的容器

    # 创建错误处理容器
    error_container = st.empty()

    # 为整个工作流程创建跟踪记录
    with trace("新闻研究", group_id=st.session_state.conversation_id):
        # 调度智能体阶段
        with message_container:
            st.write("🔍 **调度智能体**：正在制定研究方案...")

        triage_result = await Runner.run(
            triage_agent,
            f"深入研究以下主题：{topic}。研究结果将用于生成全面的研究报告。"
        )

        # 检查结果是否为ResearchPlan对象或字符串
        if hasattr(triage_result.final_output, 'topic'):
            research_plan = triage_result.final_output
            plan_display = {
                "topic": research_plan.topic,
                "search_queries": research_plan.search_queries,
                "focus_areas": research_plan.focus_areas
            }
        else:
            # 若未获取到预期输出类型，则使用备用方案
            research_plan = {
                "topic": topic,
                "search_queries": [f"研究 {topic}"],
                "focus_areas": [f"{topic} 的一般信息"]
            }
            plan_display = research_plan

        with message_container:
            st.write("📋 **研究计划**：")
            st.json(plan_display)  # 以JSON格式显示研究计划

        # 实时显示收集到的事实
        fact_placeholder = message_container.empty()

        # 定期检查新事实（增加检查次数以确保研究全面性）
        previous_fact_count = 0
        for i in range(15):
            current_facts = len(st.session_state.collected_facts)
            if current_facts > previous_fact_count:
                with fact_placeholder.container():
                    st.write("📚 **已收集事实**：")
                    for fact in st.session_state.collected_facts:
                        st.info(f"**事实**：{fact['fact']}\n\n**来源**：{fact['source']}")
                previous_fact_count = current_facts
            await asyncio.sleep(1)  # 每秒检查一次

        # 编辑智能体阶段
        with message_container:
            st.write("📝 **编辑智能体**：正在生成全面研究报告...")

        try:
            report_result = await Runner.run(
                editor_agent,
                triage_result.to_input_list()  # 将调度结果转换为输入列表
            )

            st.session_state.report_result = report_result.final_output

            with message_container:
                st.write("✅ **研究完成！报告已生成。**")

                # 预览报告片段
                if hasattr(report_result.final_output, 'report'):
                    report_preview = report_result.final_output.report[:300] + "..."  # 取前300字符
                else:
                    report_preview = str(report_result.final_output)[:300] + "..."

                st.write("📄 **报告预览**：")
                st.markdown(report_preview)
                st.write("*完整报告请查看「报告」标签页。*")

        except Exception as e:
            st.error(f"生成报告时出错：{str(e)}")
            # 若出错，显示原始智能体响应作为备用方案
            if hasattr(triage_result, 'new_items'):
                # 筛选包含内容的项目
                messages = [item for item in triage_result.new_items if hasattr(item, 'content')]
                if messages:
                    raw_content = "\n\n".join([str(m.content) for m in messages if m.content])
                    st.session_state.report_result = raw_content

                    with message_container:
                        st.write("⚠️ **研究已完成，但生成结构化报告时出现问题。**")
                        st.write("原始研究结果可在「报告」标签页查看。")

    st.session_state.research_done = True  # 标记研究完成


# 点击开始按钮后执行研究
if start_button:
    with st.spinner(f"正在研究：{user_topic}"):  # 显示加载动画
        try:
            asyncio.run(run_research(user_topic))  # 运行异步研究函数
        except Exception as e:
            st.error(f"研究过程中发生错误：{str(e)}")
            # 设置基础报告结果，确保用户获得反馈
            st.session_state.report_result = f"# {user_topic} 研究报告\n\n研究过程中不幸发生错误，请稍后重试或更换研究主题。\n\n错误详情：{str(e)}"
            st.session_state.research_done = True

# 在「报告」标签页显示结果
with tab2:
    if st.session_state.research_done and st.session_state.report_result:
        report = st.session_state.report_result

        # 处理不同类型的报告结果
        if hasattr(report, 'title'):
            # 若为结构化ResearchReport对象
            title = report.title

            # 若有大纲则显示（默认展开）
            if hasattr(report, 'outline') and report.outline:
                with st.expander("报告大纲", expanded=True):
                    for i, section in enumerate(report.outline):
                        st.markdown(f"{i + 1}. {section}")

            # 若有字数统计则显示
            if hasattr(report, 'word_count'):
                st.info(f"字数统计：{report.word_count}")

            # 以Markdown格式显示完整报告
            if hasattr(report, 'report'):
                report_content = report.report
                st.markdown(report_content)
            else:
                report_content = str(report)
                st.markdown(report_content)

            # 若有来源列表则显示
            if hasattr(report, 'sources') and report.sources:
                with st.expander("信息来源"):
                    for i, source in enumerate(report.sources):
                        st.markdown(f"{i + 1}. {source}")

            # 添加报告下载按钮
            st.download_button(
                label="下载报告",
                data=report_content,
                file_name=f"{title.replace(' ', '_')}.md",  # 文件名替换空格为下划线
                mime="text/markdown"  # 文件类型为Markdown
            )
        else:
            # 处理字符串或其他类型的响应
            report_content = str(report)
            title = user_topic.title()  # 将主题首字母大写作为标题

            st.title(f"{title}")
            st.markdown(report_content)

            # 添加报告下载按钮
            st.download_button(
                label="下载报告",
                data=report_content,
                file_name=f"{title.replace(' ', '_')}.md",
                mime="text/markdown"
            )