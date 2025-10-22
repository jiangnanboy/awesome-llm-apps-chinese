import streamlit as st
from agno.agent import Agent
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.models.anthropic import Claude
from agno.tools.newspaper4k import Newspaper4kTools
from agno.tools import Tool
import logging

logging.basicConfig(level=logging.DEBUG)

# 设置Streamlit应用
st.title("AI创业趋势分析代理 📈")
st.caption("一键获取基于您感兴趣主题的最新趋势分析和创业机会！")

topic = st.text_input("输入您创业感兴趣的领域：")
anthropic_api_key = st.sidebar.text_input("输入Anthropic API密钥", type="password")

if st.button("生成分析"):
    if not anthropic_api_key:
        st.warning("请输入所需的API密钥。")
    else:
        with st.spinner("正在处理您的请求..."):
            try:
                # 初始化Anthropic模型
                anthropic_model = Claude(id="claude-3-5-sonnet-20240620", api_key=anthropic_api_key)

                # 定义新闻收集代理 - Duckduckgo_search工具使代理能够在网上搜索信息
                search_tool = DuckDuckGoTools(search=True, news=True, fixed_max_results=5)
                news_collector = Agent(
                    name="新闻收集器",
                    role="收集关于给定主题的最新新闻文章",
                    tools=[search_tool],
                    model=anthropic_model,
                    instructions=["收集关于该主题的最新文章"],
                    show_tool_calls=True,
                    markdown=True,
                )

                # 定义摘要撰写代理
                news_tool = Newspaper4kTools(read_article=True, include_summary=True)
                summary_writer = Agent(
                    name="摘要撰写器",
                    role="总结收集到的新闻文章",
                    tools=[news_tool],
                    model=anthropic_model,
                    instructions=["提供文章的简明摘要"],
                    show_tool_calls=True,
                    markdown=True,
                )

                # 定义趋势分析代理
                trend_analyzer = Agent(
                    name="趋势分析器",
                    role="从摘要中分析趋势",
                    model=anthropic_model,
                    instructions=["识别新兴趋势和创业机会"],
                    show_tool_calls=True,
                    markdown=True,
                )

                # phidata的多代理团队设置：
                agent_team = Agent(
                    agents=[news_collector, summary_writer, trend_analyzer],
                    instructions=[
                        "首先，在DuckDuckGo上搜索与用户指定主题相关的最新新闻文章。",
                        "然后，将收集到的文章链接提供给摘要撰写器。",
                        "重要提示：您必须确保摘要撰写器收到所有要阅读的文章链接。",
                        "接下来，摘要撰写器将阅读文章并为每篇文章准备简明摘要。",
                        "总结完成后，摘要将传递给趋势分析器。",
                        "最后，趋势分析器将根据提供的摘要，以详细报告的形式识别新兴趋势和潜在的创业机会，以便任何年轻企业家都能轻松从中获得巨大价值。"
                    ],
                    show_tool_calls=True,
                    markdown=True,
                )

                # 执行工作流程
                # 步骤1：收集新闻
                news_response = news_collector.run(f"收集关于{topic}的最新新闻")
                articles = news_response.content

                # 步骤2：总结文章
                summary_response = summary_writer.run(f"总结以下文章：\n{articles}")
                summaries = summary_response.content

                # 步骤3：分析趋势
                trend_response = trend_analyzer.run(f"从以下摘要中分析趋势：\n{summaries}")
                analysis = trend_response.content

                # 显示结果 - 如果您想进一步使用，可取消下面2行的注释以获取摘要！
                # st.subheader("新闻摘要")
                # st.write(summaries)

                st.subheader("趋势分析和潜在创业机会")
                st.write(analysis)

            except Exception as e:
                st.error(f"发生错误：{e}")
else:
    st.info("输入主题和API密钥，然后点击'生成分析'开始。")