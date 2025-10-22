# Llama-3.1 与 RAG 结合的本地 Streamlit 应用

## 💻 本地 Llama-3.1 与 RAG 结合应用

这款 Streamlit 应用程序允许你通过本地运行的 Llama-3.1 模型和检索增强生成（RAG）技术与任何网页内容进行对话。该应用完全在你的电脑上运行，100% 免费且无需互联网连接。

### 功能特点

* 输入网页 URL

* 针对网页内容提问

* 借助 RAG 技术和本地运行的 Llama-3.1 模型获取准确答案

### 如何开始使用？

1. 克隆 GitHub 仓库

```
git clone https://github.com/Shubhamsaboo/awesome-llm-apps.git

cd awesome-llm-apps/rag\_tutorials/llama3.1\_local\_rag
```

2. 安装所需依赖：

```
pip install -r requirements.txt
```

3. 运行 Streamlit 应用

```
streamlit run llama3.1\_local\_rag.py
```

### 工作原理

* 应用使用 WebBaseLoader 加载网页数据，并通过 RecursiveCharacterTextSplitter 将其分割成文本块。

* 它创建 Ollama 嵌入，并使用 Chroma 构建向量存储。

* 应用搭建了一个 RAG（检索增强生成）链，根据用户的问题检索相关文档。

* 调用 Llama-3.1 模型，利用检索到的上下文生成答案。

* 应用向用户展示问题的答案。
