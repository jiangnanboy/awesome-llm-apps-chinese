import json
import tempfile
import csv
import streamlit as st
import pandas as pd
from agno.models.openai import OpenAIChat
from phi.agent.duckdb import DuckDbAgent
from agno.tools.pandas import PandasTools
import re

# 预处理并保存上传文件的函数
def preprocess_and_save(file):
    try:
        # 将上传的文件读入DataFrame
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, encoding='utf-8', na_values=['NA', 'N/A', 'missing'])
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file, na_values=['NA', 'N/A', 'missing'])
        else:
            st.error("不支持的文件格式。请上传CSV或Excel文件。")
            return None, None, None

        # 确保字符串列被正确引用
        for col in df.select_dtypes(include=['object']):
            df[col] = df[col].astype(str).replace({r'"': '""'}, regex=True)

        # 解析日期和数值列
        for col in df.columns:
            if 'date' in col.lower():
                df[col] = pd.to_datetime(df[col], errors='coerce')
            elif df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    # 转换失败则保持原样
                    pass

        # 创建临时文件保存预处理后的数据
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_path = temp_file.name
            # 将DataFrame保存到临时CSV文件，字符串字段用引号括起来
            df.to_csv(temp_path, index=False, quoting=csv.QUOTE_ALL)

        return temp_path, df.columns.tolist(), df  # 同时返回DataFrame
    except Exception as e:
        st.error(f"文件处理错误: {e}")
        return None, None, None


# Streamlit应用
st.title("📊 数据分析助手")

# API密钥侧边栏
with st.sidebar:
    st.header("API密钥")
    openai_key = st.text_input("输入您的OpenAI API密钥:", type="password")
    if openai_key:
        st.session_state.openai_key = openai_key
        st.success("API密钥已保存!")
    else:
        st.warning("请输入您的OpenAI API密钥以继续。")

# 文件上传组件
uploaded_file = st.file_uploader("上传CSV或Excel文件", type=["csv", "xlsx"])

if uploaded_file is not None and "openai_key" in st.session_state:
    # 预处理并保存上传的文件
    temp_path, columns, df = preprocess_and_save(uploaded_file)

    if temp_path and columns and df is not None:
        # 以表格形式显示上传的数据
        st.write("上传的数据:")
        st.dataframe(df)  # 使用st.dataframe显示交互式表格

        # 显示上传数据的列
        st.write("上传的列:", columns)

        # 使用临时文件路径配置语义模型
        semantic_model = {
            "tables": [
                {
                    "name": "uploaded_data",
                    "description": "包含上传的数据集。",
                    "path": temp_path,
                }
            ]
        }

        # 初始化用于SQL查询生成的DuckDbAgent
        duckdb_agent = DuckDbAgent(
            model=OpenAIChat(model="gpt-4", api_key=st.session_state.openai_key),
            semantic_model=json.dumps(semantic_model),
            tools=[PandasTools()],
            markdown=True,
            add_history_to_messages=False,  # 禁用聊天历史
            followups=False,  # 禁用后续查询
            read_tool_call_history=False,  # 禁用读取工具调用历史
            system_prompt="你是一位专业的数据分析师。生成SQL查询来解决用户的问题。只返回SQL查询，用```sql ```包裹，并给出最终答案。",
        )

        # 在会话状态中初始化代码存储
        if "generated_code" not in st.session_state:
            st.session_state.generated_code = None

        # 主查询输入组件
        user_query = st.text_area("询问有关数据的问题:")

        # 添加关于终端输出的信息提示
        st.info("💡 查看终端可获得更清晰的助手响应输出")

        if st.button("提交查询"):
            if user_query.strip() == "":
                st.warning("请输入查询内容。")
            else:
                try:
                    # 处理时显示加载动画
                    with st.spinner('正在处理您的查询...'):
                        # 从DuckDbAgent获取响应
                        response1 = duckdb_agent.run(user_query)

                        # 从RunResponse对象中提取内容
                        if hasattr(response1, 'content'):
                            response_content = response1.content
                        else:
                            response_content = str(response1)
                        response = duckdb_agent.print_response(
                            user_query,
                            stream=True,
                        )

                    # 在Streamlit中显示响应
                    st.markdown(response_content)


                except Exception as e:
                    st.error(f"DuckDbAgent生成响应时出错: {e}")
                    st.error("请尝试重新表述您的查询或检查数据格式是否正确。")
