# 📊 人工智能数据可视化代理（AI Data Visualization Agent）

这是一款基于 Streamlit 开发的应用程序，可作为您的个人数据可视化专家，其核心驱动力为大型语言模型（LLMs）。您只需上传数据集，并以自然语言提出问题 —— 该人工智能代理便会分析您的数据，生成合适的可视化图表，并通过图表、统计数据与解读说明相结合的方式，为您提供数据洞察。

## 功能特点（Features）

### 自然语言数据分析（Natural Language Data Analysis）

* 用通俗易懂的英文提出关于数据的疑问

* 即时获取可视化图表与统计分析结果

* 接收对研究发现与数据洞察的解读说明

* 支持交互式后续追问

### 智能可视化选择（Intelligent Visualization Selection）

* 自动选择合适的图表类型

* 动态生成可视化内容

* 支持统计类可视化呈现

* 可自定义图表格式与样式

### 多模型人工智能支持（Multi-Model AI Support）

* Meta-Llama 3.1 405B：适用于复杂数据分析

* DeepSeek V3：适用于获取详细数据洞察

* Qwen 2.5 7B：适用于快速数据分析

* Meta-Llama 3.3 70B：适用于高级查询场景

## 运行方法（How to Run）

按照以下步骤设置并运行该应用程序：

* 首先，请在此处获取免费的 Together AI API 密钥：[https://api.together.ai/signin](https://api.together.ai/signin)

* 在此处获取免费的 E2B API 密钥：[https://e2b.dev/](https://e2b.dev/) ；[https://e2b.dev/docs/legacy/getting-started/api-key](https://e2b.dev/docs/legacy/getting-started/api-key)

1. **克隆代码仓库（Clone the Repository）**

```
git clone https://github.com/Shubhamsaboo/awesome-llm-apps.git

cd ai\_agent\_tutorials/ai\_data\_visualisation\_agent
```

2. **安装依赖包（Install the dependencies）**

```
pip install -r requirements.txt
```

3. **运行 Streamlit 应用（Run the Streamlit app）**

```
streamlit run ai\_data\_visualisation\_agent.py
```

### 补充说明

* 文中技术术语均采用行业通用译法，如 “Streamlit”（一种 Python Web 应用框架，常用于数据科学领域）保留原名，“LLMs” 译为 “大型语言模型”；

* 代码命令部分（如`git clone`、`pip install`）保持原文格式，确保用户可直接复制执行；

* 链接地址（如 API 密钥获取链接）未做修改，以保证用户能准确访问相关页面。

