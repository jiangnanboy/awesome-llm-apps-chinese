import streamlit as st
import asyncio
import os
from together import AsyncTogether, Together

# Set up the Streamlit app
st.title("混合智能体大语言模型应用（Mixture-of-Agents LLM App）")

# Get API key from the user
together_api_key = st.text_input("输入您的Together API密钥：", type="password")

if together_api_key:
    os.environ["TOGETHER_API_KEY"] = together_api_key
    client = Together(api_key=together_api_key)
    async_client = AsyncTogether(api_key=together_api_key)

    # Define the models
    reference_models = [
        "Qwen/Qwen2-72B-Instruct",
        "Qwen/Qwen1.5-72B-Chat",
        "mistralai/Mixtral-8x22B-Instruct-v0.1",
        "databricks/dbrx-instruct",
    ]
    aggregator_model = "mistralai/Mixtral-8x22B-Instruct-v0.1"

    # Define the aggregator system prompt
    aggregator_system_prompt = """已向您提供一组来自各类开源模型的响应，这些响应均针对最新的用户查询。您的任务是将这些响应整合为一个高质量的单一响应。关键在于需批判性地评估这些响应中包含的信息，要意识到其中部分信息可能存在偏见或不准确。您的响应不应简单复制给定的答案，而需针对该指令提供经过提炼、准确且全面的回复。请确保您的响应结构清晰、逻辑连贯，并符合最高标准的准确性与可靠性。各模型响应如下："""

    # Get user input
    user_prompt = st.text_input("输入您的问题：")

    async def run_llm(model):
        """Run a single LLM call with a reference model."""
        response = await async_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0.7,
            max_tokens=512,
        )
        return model, response.choices[0].message.content

    async def main():
        results = await asyncio.gather(*[run_llm(model) for model in reference_models])

        # Display individual model responses
        st.subheader("各模型单独响应：")
        for model, response in results:
            with st.expander(f"来自{model}的响应"):
                st.write(response)

        # Aggregate responses
        st.subheader("聚合后响应：")
        finalStream = client.chat.completions.create(
            model=aggregator_model,
            messages=[
                {"role": "system", "content": aggregator_system_prompt},
                {"role": "user", "content": ",".join(response for _, response in results)},
            ],
            stream=True,
        )

        # Display aggregated response
        response_container = st.empty()
        full_response = ""
        for chunk in finalStream:
            content = chunk.choices[0].delta.content or ""
            full_response += content
            response_container.markdown(full_response + "▌")
        response_container.markdown(full_response)

    if st.button("获取答案"):
        if user_prompt:
            asyncio.run(main())
        else:
            st.warning("请输入一个问题。")

else:
    st.warning("请输入您的Together API密钥以使用此应用。")

# Add some information about the app
st.sidebar.title("关于此应用")
st.sidebar.write(
    "本应用采用“混合智能体（Mixture-of-Agents）”方案，通过多个语言模型（LLM）协作回答单个问题。"
)

st.sidebar.subheader("工作流程：")
st.sidebar.markdown(
    """
    1. 应用将您的问题发送至以下多个大语言模型（LLM）：
        - Qwen/Qwen2-72B-Instruct
        - Qwen/Qwen1.5-72B-Chat
        - mistralai/Mixtral-8x22B-Instruct-v0.1
        - databricks/dbrx-instruct
    2. 每个模型分别生成独立响应
    3. 所有响应将通过Mixtral-8x22B-Instruct-v0.1模型进行聚合处理
    4. 最终展示聚合后的响应结果
    """
)

st.sidebar.write(
    "这种方案通过整合多个AI模型的优势，能够生成更全面、更均衡的答案。"
)