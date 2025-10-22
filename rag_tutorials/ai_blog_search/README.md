# 基于 LangGraph 的智能体驱动型检索增强生成（Agentic RAG）：AI 博客搜索工具

## 概述

AI 博客搜索工具是一款**智能体驱动型检索增强生成（Agentic RAG）应用**，旨在优化从 AI 相关博客文章中检索信息的效率。该系统借助 LangChain、LangGraph 和谷歌 Gemini 模型，实现博客内容的抓取、处理与分析，为用户提供准确且符合上下文的相关答案。

## LangGraph 工作流

![LangGraph工作流](https://github.com/user-attachments/assets/07d8a6b5-f1ef-4b7e-b47a-4f14a192bd8a)

## 演示地址

[https://github.com/user-attachments/assets/cee07380-d3dc-45f4-ad26-7d944ba9c32b](https://github.com/user-attachments/assets/cee07380-d3dc-45f4-ad26-7d944ba9c32b)

## 功能特性

* **文档检索**：采用 Qdrant 作为向量数据库，基于嵌入向量（embeddings）存储并检索博客内容。

* **智能体驱动的查询处理**：通过 AI 驱动的智能体（Agent）判断查询需求 —— 需重写查询、直接回答，还是需进一步检索信息。

* **相关性评估**：利用谷歌 Gemini 模型构建自动化相关性评分系统。

* **查询优化**：对结构不完整的查询进行优化，以提升检索结果质量。

* **Streamlit 用户界面**：提供易用的交互界面，支持输入博客 URL、查询内容，并获取具有洞察力的回复。

* **基于图的工作流**：通过 LangGraph 构建结构化状态图，实现高效的决策流程。

## 所用技术栈

* **编程语言**：[Python 3.10+](https://www.python.org/downloads/release/python-31011/)

* **框架**：[LangChain](https://www.langchain.com/) 和 [LangGraph](https://langchain-ai.github.io/langgraph/tutorials/introduction/)

* **数据库**：[Qdrant](https://qdrant.tech/)

* **模型**：

  * 嵌入向量模型：[谷歌 Gemini API（embedding-001）](https://ai.google.dev/gemini-api/docs/embeddings)

  * 对话模型：[谷歌 Gemini API（gemini-2.0-flash）](https://ai.google.dev/gemini-api/docs/models/gemini#gemini-2.0-flash)

* **博客加载器**：[LangChain WebBaseLoader](https://python.langchain.com/docs/integrations/document_loaders/web_base/)

* **文档分割器**：[RecursiveCharacterTextSplitter（递归字符文本分割器）](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/recursive_text_splitter/)

* **用户界面（UI）**：[Streamlit](https://docs.streamlit.io/)

## 使用要求

1. **安装依赖包**：

```
pip install -r requirements.txt
```

2. **运行应用**：

```
streamlit run app.py
```

3. **使用步骤**：

* 在侧边栏粘贴您的谷歌 API 密钥（Google API Key）。

* 粘贴目标博客的链接。

* 输入您关于该博客文章的查询内容。

## 📬 与我联系

![握手动图](https://media.giphy.com/media/2HtWpp60NQ9CU/giphy.gif)

![codewithcharan](https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/linked-in-alt.svg)

![](https://readme-typing-svg.herokuapp.com/?font=Righteous\&size=35\&center=true\&vCenter=true\&width=500\&height=70\&duration=4000\&lines=感谢访问！👋;欢迎在领英上与我交流！;期待各类合作机会：）)

### 术语说明（补充）

* **Agentic RAG**：即 “智能体驱动型检索增强生成”，在传统 RAG（检索增强生成）基础上加入 AI 智能体，让系统能自主决策查询处理、检索策略等步骤，提升灵活性与准确性。

* **向量数据库（Vector Database）**：用于存储文本转化后的 “嵌入向量”（Embeddings），可快速匹配与查询语义相似的内容，是高效检索的核心组件。

* **Streamlit**：一款 Python 轻量级 Web 框架，专注于快速构建数据科学与 AI 相关的交互界面，无需复杂的前端开发知识。
