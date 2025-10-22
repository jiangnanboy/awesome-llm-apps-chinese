# import necessary python libraries
from agno.agent import Agent
from agno.models.xai import xAI
from agno.tools.yfinance import YFinanceTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.playground import Playground, serve_playground_app

# create the AI finance agent
agent = Agent(
    name="xAI 金融智能体",
    model = xAI(id="grok-beta"),
    tools=[DuckDuckGoTools(), YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True)],
    instructions = ["始终使用表格来展示财务 / 数值数据。对于文本数据，使用项目符号和小段落。"],
    show_tool_calls = True,
    markdown = True,
    )

# UI for finance agent
app = Playground(agents=[agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("xai_finance_agent:app", reload=True)