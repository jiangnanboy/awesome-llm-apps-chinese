# 未命名文档

## 基于 Cohere ⌘R 的 RAG 智能体系统

这是一个基于 Cohere 新模型 Command-r7b-12-2024 构建的 RAG 智能体系统，结合 Qdrant 作为向量存储、Langchain 实现 RAG 功能，以及 LangGraph 进行流程编排。该应用允许用户上传文档、针对文档提问，并获取 AI 生成的响应，必要时还会自动调用网页搜索作为补充。

## 功能特点

### 文档处理

* PDF 文档上传与处理

* 自动文本分块与嵌入

* 向量存储于 Qdrant 云服务

### 智能查询

* 基于 RAG 的文档检索

* 带阈值过滤的相似度搜索

* 当未找到相关文档时自动切换至网页搜索

* 答案来源标注

### 高级功能

* DuckDuckGo 网页搜索集成

* 用于网络研究的 LangGraph 智能体

* 上下文感知的响应生成

* 长答案摘要生成

### 模型特定功能

* Command-r7b-12-2024 模型用于聊天和 RAG

* cohere embed-english-v3.0 模型用于嵌入生成

* 来自 langgraph 的 create\_react\_agent 函数

* 用于网页搜索的 DuckDuckGoSearchRun 工具

## 前置条件

### 1. Cohere API 密钥

1. 访问[Cohere 平台](https://dashboard.cohere.ai/api-keys)

2. 注册或登录账号

3. 进入 API 密钥部分

4. 创建新的 API 密钥

### 2. Qdrant 云设置

1. 访问[Qdrant 云](https://cloud.qdrant.io/)

2. 创建账号或登录

3. 创建新集群

4. 获取凭证：

* Qdrant API 密钥：在 API 密钥部分找到

* Qdrant URL：你的集群 URL（格式：`https://xxx-xxx.aws.cloud.qdrant.io`）

## 运行方法

1. 克隆仓库：

```
git clone https://github.com/Shubhamsaboo/awesome-llm-apps.git

cd rag\_tutorials/rag\_agent\_cohere
```

2. 安装依赖：

```
pip install -r requirements.txt
```

3. 运行应用：

```
streamlit run rag\_agent\_cohere.py
```
