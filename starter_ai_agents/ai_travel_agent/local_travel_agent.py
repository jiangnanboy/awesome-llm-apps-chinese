from textwrap import dedent
from agno.agent import Agent
from agno.tools.serpapi import SerpApiTools
import streamlit as st
import re
from agno.models.ollama import Ollama
from icalendar import Calendar, Event
from datetime import datetime, timedelta


def generate_ics_content(plan_text: str, start_date: datetime = None) -> bytes:
    """
        根据旅行行程文本生成ICS日历文件。

        参数:
            plan_text: 旅行行程文本
            start_date: 行程的可选开始日期（默认为今天）

        返回:
            bytes: 作为字节的ICS文件内容
        """
    cal = Calendar()
    cal.add('prodid', '-//AI旅行规划器//github.com//')
    cal.add('version', '2.0')

    if start_date is None:
        start_date = datetime.today()

    # 将计划按天拆分
    day_pattern = re.compile(r'第 (\d+) 天[:\s]+(.*?)(?=第 \d+ 天|$)', re.DOTALL)
    days = day_pattern.findall(plan_text)

    if not days:  # 如果未找到天数模式，创建一个包含全部内容的全天事件
        event = Event()
        event.add('summary', "旅行行程")
        event.add('description', plan_text)
        event.add('dtstart', start_date.date())
        event.add('dtend', start_date.date())
        event.add("dtstamp", datetime.now())
        cal.add_component(event)
    else:
        # 处理每一天
        for day_num, day_content in days:
            day_num = int(day_num)
            current_date = start_date + timedelta(days=day_num - 1)

            # 为全天创建一个事件
            event = Event()
            event.add('summary', f"第 {day_num} 天行程")
            event.add('description', day_content.strip())

            # 设置为全天事件
            event.add('dtstart', current_date.date())
            event.add('dtend', current_date.date())
            event.add("dtstamp", datetime.now())
            cal.add_component(event)

    return cal.to_ical()


# 设置Streamlit应用
st.title("使用Llama-3.2的AI旅行规划器")
st.caption("通过使用本地Llama-3自动研究和规划个性化行程，让AI旅行规划器帮助你规划下一次冒险")

# 初始化会话状态以存储生成的行程
if 'itinerary' not in st.session_state:
    st.session_state.itinerary = None

# 从用户获取SerpAPI密钥
serp_api_key = st.text_input("输入Serp API密钥以使用搜索功能", type="password")

if serp_api_key:
    researcher = Agent(
        name="研究员",
        role="根据用户偏好搜索旅行目的地、活动和住宿",
        model=Ollama(id="llama3.2"),
        description=dedent(
            """\
        你是一名世界级的旅行研究员。给定一个旅行目的地和用户想要旅行的天数，
        生成用于查找相关旅行活动和住宿的搜索词列表。
        然后为每个词在网上搜索，分析结果，并返回10个最相关的结果。
        """
        ),
        instructions=[
            "给定一个旅行目的地和用户想要旅行的天数，首先生成3个与该目的地和天数相关的搜索词。",
            "对于每个搜索词，使用`search_google`并分析结果。",
            "从所有搜索结果中，返回10个与用户偏好最相关的结果。",
            "记住：结果的质量很重要。",
        ],
        tools=[SerpApiTools(api_key=serp_api_key)],
        add_datetime_to_instructions=True,
    )
    planner = Agent(
        name="规划师",
        role="根据用户偏好和研究结果生成行程草案",
        model=Ollama(id="llama3.2"),
        description=dedent(
            """\
        你是一名高级旅行规划师。给定一个旅行目的地、用户想要旅行的天数以及研究结果列表，
        你的目标是生成满足用户需求和偏好的行程草案。
        """
        ),
        instructions=[
            "给定一个旅行目的地、用户想要旅行的天数以及研究结果列表，生成包含建议活动和住宿的行程草案。",
            "确保行程结构合理、信息丰富且具有吸引力。",
            "确保提供有细微差别且平衡的行程，尽可能引用事实。",
            "记住：行程的质量很重要。",
            "注重清晰度、连贯性和整体质量。",
            "绝不编造事实或抄袭。始终提供适当的出处。",
        ],
        add_datetime_to_instructions=True,
    )

    # 用户目的地和旅行天数的输入字段
    destination = st.text_input("你想去哪里？")
    num_days = st.number_input("你想旅行多少天？", min_value=1, max_value=30, value=7)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("生成行程"):
            with st.spinner("处理中..."):
                # 从助手获取响应
                response = planner.run(f"{destination}，{num_days}天", stream=False)
                # 将响应存储在会话状态中
                st.session_state.itinerary = response.content
                st.write(response.content)

    # 只有当有行程时才显示下载按钮
    with col2:
        if st.session_state.itinerary:
            # 生成ICS文件
            ics_content = generate_ics_content(st.session_state.itinerary)

            # 提供文件下载
            st.download_button(
                label="将行程下载为日历文件（.ics）",
                data=ics_content,
                file_name="travel_itinerary.ics",
                mime="text/calendar"
            )