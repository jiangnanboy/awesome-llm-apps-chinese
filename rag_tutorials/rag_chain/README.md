# 医药查询系统（PharmaQuery）

## 概述（Overview）

医药查询系统（PharmaQuery）是一款先进的医药领域洞察检索系统，旨在帮助用户从医药领域的研究论文和文档中获取有价值的洞察信息。

## 演示（Demo）

[https://github.com/user-attachments/assets/c12ee305-86fe-4f71-9219-57c7f438f291](https://github.com/user-attachments/assets/c12ee305-86fe-4f71-9219-57c7f438f291)

## 功能（Features）

* **自然语言查询（Natural Language Querying）**：针对医药行业提出复杂问题，即可获得简洁、准确的答案。

* **自定义数据库（Custom Database）**：上传您自己的研究文档，以扩充检索系统的知识库。

* **相似度搜索（Similarity Search）**：利用人工智能嵌入技术，检索与您的查询最相关的文档。

* **Streamlit 界面（Streamlit Interface）**：具备用户友好的界面，支持查询操作和文档上传。

## 所用技术（Technologies Used）

* **编程语言（Programming Language）**：[Python 3.10+](https://www.python.org/downloads/release/python-31011/)

* **框架（Framework）**：[LangChain](https://www.langchain.com/)（一款用于构建大语言模型应用的开发框架）

* **数据库（Database）**：[ChromaDB](https://www.trychroma.com/)（轻量级向量数据库，适用于存储和检索 AI 嵌入数据）

* **模型（Models）**：

  * 嵌入模型（Embeddings）：[Google Gemini API（embedding-001）](https://ai.google.dev/gemini-api/docs/embeddings)

  * 对话模型（Chat）：[Google Gemini API（gemini-1.5-pro）](https://ai.google.dev/gemini-api/docs/models/gemini#gemini-1.5-pro)

* **PDF 处理（PDF Processing）**：[PyPDFLoader](https://python.langchain.com/docs/integrations/document_loaders/pypdfloader/)（LangChain 中的 PDF 文档加载工具，用于提取 PDF 内容）

* **文档分割器（Document Splitter）**：[SentenceTransformersTokenTextSplitter](https://python.langchain.com/api_reference/text_splitters/sentence_transformers/langchain_text_splitters.sentence_transformers.SentenceTransformersTokenTextSplitter.html)（基于 SentenceTransformers 的文本分割工具，按 token 长度拆分文档以适配模型输入要求）

## 使用要求（Requirements）

1. **安装依赖（Install Dependencies）**：

```
pip install -r requirements.txt
```

1. **运行应用（Run the Application）**：

```
streamlit run app.py
```

1. **使用应用（Use the Application）**：

* 在侧边栏中粘贴您的 Google API 密钥（Google API Key）。

* 在主界面中输入您的查询内容。

* （可选操作）在侧边栏上传研究论文，以进一步扩充数据库。

## 📬 与我联系（Connect With Me）

![握手动图（handshake gif）](https://media.giphy.com/media/2HtWpp60NQ9CU/giphy.gif)

![codewithcharan（LinkedIn账号）](https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/linked-in-alt.svg)
