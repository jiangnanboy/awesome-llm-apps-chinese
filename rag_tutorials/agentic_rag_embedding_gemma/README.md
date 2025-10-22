# 🔥 基于 EmbeddingGemma 的智能体驱动型检索增强生成（Agentic RAG）

本 Streamlit 应用展示了一个智能体驱动型检索增强生成（RAG）智能体，其中使用谷歌的 EmbeddingGemma 生成嵌入向量，以 Llama 3.2 作为语言模型，所有组件均通过 Ollama 在本地运行。

## 功能特点

* **本地 AI 模型**：采用 EmbeddingGemma 生成向量嵌入，使用 Llama 3.2 进行文本生成

* **PDF 知识库**：可动态添加 PDF 文件链接，用于构建知识库

* **向量搜索**：借助 LanceDB 实现高效的相似性搜索

* **交互式界面**：简洁美观的 Streamlit 界面，支持添加数据源与发起查询

* **流式响应**：实时生成响应内容，同时可查看工具调用过程

## 如何开始使用？

1. 克隆 GitHub 仓库

```
git clone https://github.com/Shubhamsaboo/awesome-llm-apps.git

cd awesome-llm-apps/rag\_tutorials/agentic\_rag\_embedding\_gemma
```

2. 安装所需依赖：

```
pip install -r requirements.txt
```

3. 确保已安装 Ollama 并启动，且所需模型已准备就绪：

* 拉取模型：执行命令 `ollama pull embeddinggemma:latest` 和 `ollama pull llama3.2:latest`

* 若 Ollama 服务未启动，请先启动服务

4. 运行 Streamlit 应用：

```
streamlit run agentic\_rag\_embeddinggemma.py
```

（注：应用文件位于根目录下）

1. 打开浏览器，访问系统提供的 URL（通常为 [http://localhost:8501](http://localhost:8501)），即可与该 RAG 智能体进行交互。

## 工作原理

1. **知识库搭建**：在侧边栏中添加 PDF 文件链接，加载文档并建立索引。

2. **嵌入向量生成**：EmbeddingGemma 生成向量嵌入，为语义搜索提供支持。

3. **查询处理**：对用户查询生成嵌入向量，并在知识库中进行检索匹配。

4. **响应生成**：Llama 3.2 根据检索到的上下文信息生成回答。

5. **工具集成**：智能体通过调用搜索工具获取相关信息。

## 环境要求

* Python 3.8 及以上版本

* 已安装并启动 Ollama

* 所需模型：`embeddinggemma:latest`、`llama3.2:latest`

## 使用的技术栈

* **Agno**：用于构建 AI 智能体的框架

* **Streamlit**：Web 应用开发框架

* **LanceDB**：向量数据库

* **Ollama**：本地大语言模型（LLM）服务器

* **EmbeddingGemma**：谷歌推出的嵌入模型

* **Llama 3.2**：Meta（元宇宙）推出的语言模型

