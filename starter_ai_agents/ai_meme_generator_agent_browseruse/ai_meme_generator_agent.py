import asyncio
import streamlit as st
from browser_use import Agent, SystemPrompt
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
import re


async def generate_meme(query: str, model_choice: str, api_key: str) -> None:
    # 根据用户选择初始化相应的大语言模型
    if model_choice == "Claude":
        llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            api_key=api_key
        )
    elif model_choice == "Deepseek":
        llm = ChatOpenAI(
            base_url='https://api.deepseek.com/v1',
            model='deepseek-chat',
            api_key=api_key,
            temperature=0.3
        )
    else:  # OpenAI
        llm = ChatOpenAI(
            model="gpt-4o",
            api_key=api_key,
            temperature=0.0
        )

    task_description = (
        "你是一位表情包生成专家。给定一个查询，你需要为其生成一个表情包。\n"
        "1. 访问 https://imgflip.com/memetemplates \n"
        "2. 点击中间的搜索栏，仅搜索该查询中的一个主要动作动词（例如'欺负'、'大笑'、'哭泣'）：'{0}'\n"
        "3. 选择任何一个隐喻上符合表情包主题的模板：'{0}'\n"
        "   点击模板下方的'Add Caption'按钮\n"
        "4. 编写与'{0}'相关的顶部文字（铺垫/背景）和底部文字（笑点/结果）。\n"
        "5. 检查预览，确保它有趣且是有意义的表情包。必要时直接调整文字。\n"
        "6. 查看表情包及其上的文字，如果不合理，请重新尝试，在文本框中填入不同的文字。\n"
        "7. 点击'Generate meme'按钮生成表情包\n"
        "8. 复制图片链接并将其作为输出\n"
    ).format(query)

    agent = Agent(
        task=task_description,
        llm=llm,
        max_actions_per_step=5,
        max_failures=25,
        use_vision=(model_choice != "Deepseek")  # Deepseek不使用视觉功能
    )

    history = await agent.run()

    # 从代理历史中提取最终结果
    final_result = history.final_result()

    # 使用正则表达式从结果中找到表情包URL
    url_match = re.search(r'https://imgflip\.com/i/(\w+)', final_result)
    if url_match:
        meme_id = url_match.group(1)
        return f"https://i.imgflip.com/{meme_id}.jpg"
    return None


def main():
    # 自定义CSS样式

    st.title("🥸 AI 表情包生成代理 - 浏览器使用")
    st.info("这个AI浏览器代理通过浏览器自动化，根据你的输入生成表情包。请输入你的API密钥并描述你想要生成的表情包。")

    # 侧边栏配置
    with st.sidebar:
        st.markdown('<p class="sidebar-header">⚙️ 模型配置</p>', unsafe_allow_html=True)

        # 模型选择
        model_choice = st.selectbox(
            "选择AI模型",
            ["Claude", "Deepseek", "OpenAI"],
            index=0,
            help="选择用于生成表情包的大语言模型"
        )

        # 根据模型选择输入相应的API密钥
        api_key = ""
        if model_choice == "Claude":
            api_key = st.text_input("Claude API密钥", type="password",
                                    help="从https://console.anthropic.com获取你的API密钥")
        elif model_choice == "Deepseek":
            api_key = st.text_input("Deepseek API密钥", type="password",
                                    help="从https://platform.deepseek.com获取你的API密钥")
        else:
            api_key = st.text_input("OpenAI API密钥", type="password",
                                    help="从https://platform.openai.com获取你的API密钥")

    # 主内容区域
    st.markdown('<p class="header-text">🎨 描述你的表情包概念</p>', unsafe_allow_html=True)

    query = st.text_input(
        "表情包创意输入",
        placeholder="示例：'伊利亚的SSI静静地看着OpenAI与Deepseek的争论，同时勤奋地研究ASI'",
        label_visibility="collapsed"
    )

    if st.button("生成表情包 🚀"):
        if not api_key:
            st.warning(f"请提供{model_choice}的API密钥")
            st.stop()
        if not query:
            st.warning("请输入表情包创意")
            st.stop()

        with st.spinner(f"🧠 {model_choice}正在生成你的表情包..."):
            try:
                meme_url = asyncio.run(generate_meme(query, model_choice, api_key))

                if meme_url:
                    st.success("✅ 表情包生成成功！")
                    st.image(meme_url, caption="生成的表情包预览", use_container_width=True)
                    st.markdown(f"""
                        **直接链接：** [在ImgFlip中打开]({meme_url})  
                        **嵌入URL：** `{meme_url}`
                    """)
                else:
                    st.error("❌ 生成表情包失败。请尝试使用不同的提示词重试。")

            except Exception as e:
                st.error(f"错误：{str(e)}")
                st.info("💡 如果使用OpenAI，请确保你的账户有权限访问GPT-4o")


if __name__ == '__main__':
    main()