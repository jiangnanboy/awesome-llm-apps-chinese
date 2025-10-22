import streamlit as st
from duckduckgo_search import DDGS
from swarm import Swarm, Agent
from datetime import datetime
from dotenv import load_dotenv

# 加载环境变量（需提前在.env文件中配置相关密钥等信息）
load_dotenv()
# 指定使用的大模型（此处为llama3.2最新版本）
MODEL = "llama3.2:latest"
# 初始化Swarm客户端（用于管理智能体交互）
client = Swarm()

# 配置Streamlit页面基础信息
st.set_page_config(page_title="AI新闻处理器", page_icon="📰")
# 设置页面标题
st.title("📰 新闻简讯智能体（News Inshorts Agent）")


def search_news(topic):
    """通过DuckDuckGo搜索引擎获取新闻文章"""
    with DDGS() as ddg:
        # 搜索指定主题的当月新闻，最多返回3条结果
        results = ddg.text(f"{topic} 新闻 {datetime.now().strftime('%Y-%m')}", max_results=3)
        if results:
            # 将搜索结果按"标题+链接+摘要"格式结构化拼接
            news_results = "\n\n".join([
                f"标题：{result['title']}\n链接：{result['href']}\n摘要：{result['body']}"
                for result in results
            ])
            return news_results
        # 无结果时返回提示信息
        return f"未找到关于「{topic}」的新闻。"


# 创建专业智能体1：新闻搜索智能体
search_agent = Agent(
    name="新闻搜索员（News Searcher）",
    instructions="""
    你是一名专业的新闻搜索专员，核心任务如下：
    1. 针对给定主题，搜索最相关、最新鲜的新闻内容
    2. 确保搜索结果来自具有公信力的正规信息源
    3. 以结构化格式返回原始搜索结果（包含标题、链接、摘要）
    """,
    # 绑定新闻搜索函数
    functions=[search_news],
    # 指定使用的大模型
    model=MODEL
)

# 创建专业智能体2：新闻整合智能体
synthesis_agent = Agent(
    name="新闻整合员（News Synthesizer）",
    instructions="""
    你是一名资深新闻整合专家，核心任务如下：
    1. 深入分析所提供的原始新闻文章内容
    2. 识别新闻中的核心主题与关键信息点
    3. 融合多个信息源的内容，避免重复表述
    4. 生成内容全面且简洁凝练的整合文本
    5. 聚焦客观事实，保持新闻报道的中立性
    6. 采用清晰、专业的书面表达风格
    请针对核心要点生成2-3个段落的整合内容。
    """,
    model=MODEL
)

# 创建专业智能体3：新闻摘要智能体
summary_agent = Agent(
    name="新闻摘要员（News Summarizer）",
    instructions="""
    你是一名专业新闻摘要师，需融合美联社（AP）与路透社的清晰风格，同时兼顾数字时代的简洁性。

    核心任务要求：
    1. 核心信息提取：
       - 以最具新闻价值的进展作为开头
       - 包含关键利益相关方及其具体行动
       - 若相关，需补充重要数据或数字信息
       - 解释当前事件的重要性与时效性
       - 提及短期可能产生的影响或后续走向

    2. 风格规范：
       - 使用有力、主动的动词（避免被动语态）
       - 表述需具体明确，避免笼统模糊
       - 保持新闻报道的客观性（不加入主观评价）
       - 每句话都需有信息价值，避免冗余
       - 若涉及专业术语，需进行通俗解释

    格式要求：生成1段250-400字的文本，既要传递完整信息，又要具备可读性。
   结构模板：[核心新闻事件] + [关键细节/数据] + [事件意义/后续展望]

    思考重点：需回答三个问题——发生了什么？为何重要？影响是什么？

    重要提醒：仅提供摘要文本，不得包含引导语、标签或元文本（如“以下是摘要”“按AP风格撰写如下”等）。
    直接以新闻内容开头。
    """,
    model=MODEL
)


def process_news(topic):
    """新闻处理主流程（串联三个智能体完成搜索→整合→摘要）"""
    # 显示处理状态面板（默认展开）
    with st.status("正在处理新闻...", expanded=True) as status:
        # 第一步：调用搜索智能体获取原始新闻
        status.write("🔍 正在搜索新闻...")
        search_response = client.run(
            agent=search_agent,
            messages=[{"role": "user", "content": f"查找关于「{topic}」的近期新闻"}]
        )
        # 提取搜索结果中的文本内容（取最后一条消息，即智能体的回复）
        raw_news = search_response.messages[-1]["content"]

        # 第二步：调用整合智能体处理原始新闻
        status.write("🔄 正在整合信息...")
        synthesis_response = client.run(
            agent=synthesis_agent,
            messages=[{"role": "user", "content": f"请整合以下新闻文章：\n{raw_news}"}]
        )
        synthesized_news = synthesis_response.messages[-1]["content"]

        # 第三步：调用摘要智能体生成最终摘要
        status.write("📝 正在生成摘要...")
        summary_response = client.run(
            agent=summary_agent,
            messages=[{"role": "user", "content": f"请为以下整合内容生成摘要：\n{synthesized_news}"}]
        )

        # 返回三个阶段的结果（原始新闻、整合内容、最终摘要）
        return raw_news, synthesized_news, summary_response.messages[-1]["content"]


# 构建用户交互界面（UI）
# 新闻主题输入框（默认值为“人工智能”）
topic = st.text_input("请输入新闻主题：", value="人工智能")
# 主功能按钮（点击触发新闻处理流程）
if st.button("处理新闻", type="primary"):
    # 验证输入不为空
    if topic:
        try:
            # 执行完整新闻处理流程
            raw_news, synthesized_news, final_summary = process_news(topic)
            # 显示最终结果（标题+摘要）
            st.header(f"📝 新闻摘要：{topic}")
            st.markdown(final_summary)
        # 捕获并显示异常（如网络错误、API调用失败等）
        except Exception as e:
            st.error(f"处理过程中发生错误：{str(e)}")
    # 输入为空时提示用户
    else:
        st.error("请输入新闻主题！")