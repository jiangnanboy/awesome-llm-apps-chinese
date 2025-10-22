import os
import json
import re
import sys
import io
import contextlib
import warnings
from typing import Optional, List, Any, Tuple
from PIL import Image
import streamlit as st
import pandas as pd
import base64
from io import BytesIO
from together import Together
from e2b_code_interpreter import Sandbox

# 忽略pydantic模块的UserWarning警告
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")


# 用于匹配pattern = re.compile(r"```python\n(.*?)\n```", re.DOTALL)

def code_interpret(e2b_code_interpreter: Sandbox, code: str) -> Optional[List[Any]]:
    with st.spinner('在E2B沙箱中执行代码...'):
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                exec = e2b_code_interpreter.run_code(code)

        # 输出错误信息
        if stderr_capture.getvalue():
            print("[代码解释器警告/错误]", file=sys.stderr)
            print(stderr_capture.getvalue(), file=sys.stderr)

        # 输出标准输出信息
        if stdout_capture.getvalue():
            print("[代码解释器输出]", file=sys.stdout)
            print(stdout_capture.getvalue(), file=sys.stdout)

        # 检查执行错误
        if exec.error:
            print(f"[代码解释器错误] {exec.error}", file=sys.stderr)
            return None
        return exec.results


def match_code_blocks(llm_response: str) -> str:
    match = pattern.search(llm_response)
    if match:
        code = match.group(1)
        return code
    return ""


def chat_with_llm(e2b_code_interpreter: Sandbox, user_message: str, dataset_path: str) -> Tuple[
    Optional[List[Any]], str]:
    # 更新系统提示以包含数据集路径信息
    system_prompt = f"""你是一名Python数据科学家和数据可视化专家。你会收到一个位于'{dataset_path}'路径的数据集以及用户的查询。
你需要分析该数据集并通过响应回答用户的查询，同时运行Python代码来解决问题。
重要提示：读取CSV文件时，请在代码中始终使用数据集路径变量'{dataset_path}'。"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    with st.spinner('从Together AI大语言模型获取响应...'):
        client = Together(api_key=st.session_state.together_api_key)
        response = client.chat.completions.create(
            model=st.session_state.model_name,
            messages=messages,
        )

        response_message = response.choices[0].message
        python_code = match_code_blocks(response_message.content)

        if python_code:
            code_interpreter_results = code_interpret(e2b_code_interpreter, python_code)
            return code_interpreter_results, response_message.content
        else:
            st.warning(f"未能从模型响应中匹配到任何Python代码")
            return None, response_message.content


def upload_dataset(code_interpreter: Sandbox, uploaded_file) -> str:
    dataset_path = f"./{uploaded_file.name}"

    try:
        code_interpreter.files.write(dataset_path, uploaded_file)
        return dataset_path
    except Exception as error:
        st.error(f"文件上传过程中出错：{error}")
        raise error


def main():
    """主Streamlit应用程序。"""
    st.title("📊 AI数据可视化助手")
    st.write("上传你的数据集并提出相关问题！")

    # 初始化会话状态变量
    if 'together_api_key' not in st.session_state:
        st.session_state.together_api_key = ''
    if 'e2b_api_key' not in st.session_state:
        st.session_state.e2b_api_key = ''
    if 'model_name' not in st.session_state:
        st.session_state.model_name = ''

    with st.sidebar:
        st.header("API密钥和模型配置")
        st.session_state.together_api_key = st.sidebar.text_input("Together AI API密钥", type="password")
        st.sidebar.info("💡 每个人都可以在Together AI - 人工智能加速云平台获得1美元的免费额度")
        st.sidebar.markdown("[获取Together AI API密钥](https://api.together.ai/signin)")

        st.session_state.e2b_api_key = st.sidebar.text_input("输入E2B API密钥", type="password")
        st.sidebar.markdown("[获取E2B API密钥](https://e2b.dev/docs/legacy/getting-started/api-key)")

        # 添加模型选择下拉框
        model_options = {
            "Meta-Llama 3.1 405B": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            "深度求索V3": "deepseek-ai/DeepSeek-V3",
            "通义千问2.5 7B": "Qwen/Qwen2.5-7B-Instruct-Turbo",
            "Meta-Llama 3.3 70B": "meta-llama/Llama-3.3-70B-Instruct-Turbo"
        }
        st.session_state.model_name = st.selectbox(
            "选择模型",
            options=list(model_options.keys()),
            index=0  # 默认选择第一个选项
        )
        st.session_state.model_name = model_options[st.session_state.model_name]

    uploaded_file = st.file_uploader("选择一个CSV文件", type="csv")

    if uploaded_file is not None:
        # 显示数据集，带切换选项
        df = pd.read_csv(uploaded_file)
        st.write("数据集：")
        show_full = st.checkbox("显示完整数据集")
        if show_full:
            st.dataframe(df)
        else:
            st.write("预览（前5行）：")
            st.dataframe(df.head())
        # 查询输入
        query = st.text_area("你想了解关于你的数据的哪些信息？",
                             "你能比较不同类别之间两人的平均费用吗？")

        if st.button("分析"):
            if not st.session_state.together_api_key or not st.session_state.e2b_api_key:
                st.error("请在侧边栏中输入两个API密钥。")
            else:
                with Sandbox(api_key=st.session_state.e2b_api_key) as code_interpreter:
                    # 上传数据集
                    dataset_path = upload_dataset(code_interpreter, uploaded_file)

                    # 将dataset_path传递给chat_with_llm
                    code_results, llm_response = chat_with_llm(code_interpreter, query, dataset_path)

                    # 显示大语言模型的文本响应
                    st.write("AI响应：")
                    st.write(llm_response)

                    # 显示结果/可视化内容
                    if code_results:
                        for result in code_results:
                            if hasattr(result, 'png') and result.png:  # 检查是否有PNG数据
                                # 解码base64编码的PNG数据
                                png_data = base64.b64decode(result.png)

                                # 将PNG数据转换为图像并显示
                                image = Image.open(BytesIO(png_data))
                                st.image(image, caption="生成的可视化结果", use_container_width=False)
                            elif hasattr(result, 'figure'):  # 对于matplotlib图表
                                fig = result.figure  # 提取matplotlib图表
                                st.pyplot(fig)  # 使用st.pyplot显示
                            elif hasattr(result, 'show'):  # 对于plotly图表
                                st.plotly_chart(result)
                            elif isinstance(result, (pd.DataFrame, pd.Series)):
                                st.dataframe(result)
                            else:
                                st.write(result)


if __name__ == "__main__":
    main()