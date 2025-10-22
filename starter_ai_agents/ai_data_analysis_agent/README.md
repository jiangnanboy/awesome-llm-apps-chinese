# 📊 人工智能数据分析代理（AI Data Analysis Agent）

该人工智能数据分析代理基于 Agno Agent 框架和 OpenAI 的 gpt-4o 模型构建而成。此代理可帮助用户通过自然语言查询来分析其数据（包括 CSV、Excel 文件），其核心驱动力来自 OpenAI 的语言模型与用于高效数据处理的 DuckDB 数据库 —— 无论用户是否具备 SQL 专业知识，都能轻松开展数据分析工作。

## 功能特点（Features）

* 📤 **文件上传支持**：

  * 支持上传 CSV 和 Excel 文件

  * 自动检测数据类型并推断数据模式（schema）

  * 兼容多种文件格式

* 💬 **自然语言查询**：

  * 将自然语言问题转化为 SQL 查询语句

  * 即时获取关于数据的分析答案

  * 无需掌握 SQL 知识

* 🔍 **高级分析能力**：

  * 执行复杂的数据聚合操作

  * 对数据进行筛选与排序

  * 生成统计摘要

  * 创建数据可视化图表

* 🎯 **交互式用户界面（UI）**：

  * 简洁易用的 Streamlit 界面（Streamlit 是一款用于快速构建数据应用的 Python 框架）

  * 实时查询处理

  * 清晰的结果展示形式

## 运行方法（How to Run）

1. **环境配置（Setup Environment）**

```
\# 克隆代码仓库

git clone https://github.com/Shubhamsaboo/awesome-llm-apps.git

cd awesome-llm-apps/starter\_ai\_agents/ai\_data\_analysis\_agent

\# 安装依赖包

pip install -r requirements.txt
```

1. **配置 API 密钥（Configure API Keys）**

* 从[OpenAI 平台](https://platform.openai.com)获取 OpenAI API 密钥

1. **启动应用程序（Run the Application）**


```
streamlit run ai\_data\_analyst.py
```

## 使用说明（Usage）


1. 通过上述命令启动应用程序

2. 在 Streamlit 的侧边栏中输入你的 OpenAI API 密钥

3. 通过 Streamlit 界面上传你的 CSV 或 Excel 文件

4. 用自然语言提出关于数据的问题

5. 查看分析结果及生成的数据可视化图表
